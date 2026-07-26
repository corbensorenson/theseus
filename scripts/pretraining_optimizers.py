#!/usr/bin/env python3
"""Canonical MLX optimizer implementations for pre-training candidates."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


class OptimizerContractFault(ValueError):
    pass


OPTIMIZER_IDS = {
    "adafactor_mlx",
    "adam_mini_mlx",
    "adamw_mlx",
    "adamw_bfloat16_moments_mlx",
    "ademamix_mlx",
    "muon_mlx",
    "schedule_free_adamw_mlx",
}


ADAM_MINI_EMBEDDING_NAMES = {"embed", "embd", "embedding", "wte"}
ADAM_MINI_OUTPUT_NAMES = {"lm_head", "output", "final_layer"}
ADAM_MINI_QUERY_KEY_NAMES = {
    "k_proj",
    "q_proj",
    "wq",
    "wk",
    "query",
    "key",
}
ADAM_MINI_VALUE_NAMES = {"v_proj", "wv", "value"}
ADAM_MINI_ATTENTION_OUTPUT_NAMES = {
    "o_proj",
    "out_proj",
    "wo",
    "attn.proj",
}
ADAM_MINI_MLP_NAMES = {"feed_forward", "linear", "mlp"}


def ademamix_alpha_for_step(
    step: int, *, alpha_final: float, warmup_steps: int
) -> float:
    """Official linear AdEMAMix alpha warmup, exposed for independent tests."""

    if step < 0 or warmup_steps < 0 or alpha_final < 0.0:
        raise OptimizerContractFault("ademamix_alpha_schedule_invalid")
    if not warmup_steps or step >= warmup_steps:
        return float(alpha_final)
    return float(alpha_final) * float(step) / float(warmup_steps)


def ademamix_beta3_for_step(
    step: int,
    *,
    beta1: float,
    beta3_final: float,
    warmup_steps: int,
) -> float:
    """Official half-life-linear AdEMAMix beta3 warmup."""

    if (
        step < 0
        or warmup_steps < 0
        or not 0.0 <= beta1 < 1.0
        or not 0.0 <= beta3_final < 1.0
    ):
        raise OptimizerContractFault("ademamix_beta3_schedule_invalid")
    if not warmup_steps or step >= warmup_steps:
        return float(beta3_final)

    def half_life(beta: float) -> float:
        return math.log(0.5) / math.log(beta + 1e-8) - 1.0

    fraction = float(step) / float(warmup_steps)
    interpolated = (
        (1.0 - fraction) * half_life(beta1)
        + fraction * half_life(beta3_final)
    )
    return math.pow(0.5, 1.0 / (interpolated + 1.0))


def adam_mini_partition(path: str, value: Any | None = None) -> str:
    """Return the official Adam-mini v1.1 partition adapted to Theseus names."""

    del value
    name = path.lower()
    if "bias" in name:
        return "adam"
    if any(fragment in name for fragment in ADAM_MINI_QUERY_KEY_NAMES):
        return "head"
    if (
        any(fragment in name for fragment in ADAM_MINI_EMBEDDING_NAMES)
        or any(fragment in name for fragment in ADAM_MINI_OUTPUT_NAMES)
        or any(fragment in name for fragment in ADAM_MINI_VALUE_NAMES)
        or any(fragment in name for fragment in ADAM_MINI_MLP_NAMES)
        or any(fragment in name for fragment in ADAM_MINI_ATTENTION_OUTPUT_NAMES)
    ):
        return "row"
    if "norm" in name or ".ln" in name:
        return "block_no_decay"
    return "block"


def optimizer_contract_digest(config: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def muon_hidden_matrix_filter(path: str, value: Any) -> bool:
    """Route only hidden two-dimensional weights to Muon.

    Embeddings, vocabulary readouts, auxiliary classifiers, norms, and biases
    remain on AdamW as recommended by the MLX Muon implementation.
    """

    excluded = (
        "embedding",
        "output",
        "classifier",
        "pointer",
        "register",
        "norm",
        "bias",
    )
    return int(getattr(value, "ndim", 0)) == 2 and not any(
        fragment in path for fragment in excluded
    )


def build_optimizer(
    optimizer_id: str,
    *,
    learning_rate: Any,
    weight_decay: float,
    optim: Any,
    mx: Any,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    warmup_steps: int = 0,
    adamw_bias_correction: bool = False,
    muon_learning_rate: float | None = None,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    adafactor_eps1: float = 1e-30,
    adafactor_eps2: float = 1e-3,
    adafactor_clip_threshold: float = 1.0,
    adafactor_decay_rate: float = -0.8,
    adafactor_parameter_scale: bool = True,
    adafactor_relative_step: bool = False,
    ademamix_beta3: float = 0.9999,
    ademamix_alpha: float = 8.0,
    ademamix_beta3_warmup_steps: int = 0,
    ademamix_alpha_warmup_steps: int = 0,
    adam_mini_dim: int | None = None,
    adam_mini_num_heads: int | None = None,
    adam_mini_num_kv_heads: int | None = None,
) -> Any:
    if optimizer_id not in OPTIMIZER_IDS:
        raise OptimizerContractFault(f"optimizer_unknown:{optimizer_id}")
    if weight_decay < 0.0 or eps <= 0.0:
        raise OptimizerContractFault("optimizer_numeric_contract_invalid")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise OptimizerContractFault("optimizer_beta_contract_invalid")
    if warmup_steps < 0:
        raise OptimizerContractFault("optimizer_warmup_invalid")
    if optimizer_id == "ademamix_mlx":
        if (
            not 0.0 <= ademamix_beta3 < 1.0
            or ademamix_alpha < 0.0
            or ademamix_beta3_warmup_steps < 0
            or ademamix_alpha_warmup_steps < 0
        ):
            raise OptimizerContractFault("ademamix_numeric_contract_invalid")
        return AdEMAMix(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            beta3=float(ademamix_beta3),
            alpha=float(ademamix_alpha),
            beta3_warmup_steps=int(ademamix_beta3_warmup_steps),
            alpha_warmup_steps=int(ademamix_alpha_warmup_steps),
            eps=eps,
            weight_decay=weight_decay,
            optim=optim,
            mx=mx,
        )
    if optimizer_id == "adam_mini_mlx":
        if (
            adam_mini_dim is None
            or adam_mini_num_heads is None
            or int(adam_mini_dim) <= 0
            or int(adam_mini_num_heads) <= 0
        ):
            raise OptimizerContractFault(
                "adam_mini_requires_content_bound_model_dimensions"
            )
        num_kv_heads = (
            int(adam_mini_num_kv_heads)
            if adam_mini_num_kv_heads is not None
            else int(adam_mini_num_heads)
        )
        if (
            num_kv_heads <= 0
            or int(adam_mini_num_heads) % num_kv_heads
            or (int(adam_mini_dim) * int(adam_mini_dim))
            % int(adam_mini_num_heads)
        ):
            raise OptimizerContractFault("adam_mini_dimension_contract_invalid")
        return AdamMini(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            dim=int(adam_mini_dim),
            num_heads=int(adam_mini_num_heads),
            num_kv_heads=num_kv_heads,
            optim=optim,
            mx=mx,
        )
    if optimizer_id == "adamw_mlx":
        return optim.AdamW(
            learning_rate=learning_rate,
            betas=[beta1, beta2],
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=adamw_bias_correction,
        )
    if optimizer_id == "adamw_bfloat16_moments_mlx":
        return AdamWWithBFloat16Moments(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=adamw_bias_correction,
            optim=optim,
            mx=mx,
        )
    if optimizer_id == "adafactor_mlx":
        if (
            adafactor_eps1 <= 0.0
            or adafactor_eps2 <= 0.0
            or adafactor_clip_threshold <= 0.0
            or adafactor_decay_rate >= 0.0
        ):
            raise OptimizerContractFault("adafactor_numeric_contract_invalid")
        optimizer = optim.Adafactor(
            learning_rate=learning_rate,
            eps=(float(adafactor_eps1), float(adafactor_eps2)),
            clip_threshold=float(adafactor_clip_threshold),
            decay_rate=float(adafactor_decay_rate),
            beta_1=None,
            weight_decay=weight_decay,
            scale_parameter=bool(adafactor_parameter_scale),
            relative_step=bool(adafactor_relative_step),
            warmup_init=False,
        )
        optimizer.theseus_policy_id = "adafactor_factored_second_moment_v1"
        optimizer.factored_matrix_threshold_ndim = 2
        optimizer.vector_scalar_fallback = "unfactored_second_moment"
        optimizer.parameter_scale_policy = (
            "max_eps2_parameter_rms"
            if adafactor_parameter_scale
            else "disabled"
        )
        optimizer.update_clip_threshold = float(adafactor_clip_threshold)
        return optimizer
    if optimizer_id == "muon_mlx":
        if callable(learning_rate):
            raise OptimizerContractFault(
                "muon_candidate_requires_content_bound_scalar_learning_rate"
            )
        muon = optim.Muon(
            learning_rate=(
                float(muon_learning_rate)
                if muon_learning_rate is not None
                else learning_rate
            ),
            momentum=float(muon_momentum),
            weight_decay=weight_decay,
            nesterov=True,
            ns_steps=int(muon_ns_steps),
        )
        fallback = optim.AdamW(
            learning_rate=learning_rate,
            betas=[beta1, beta2],
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=adamw_bias_correction,
        )
        return optim.MultiOptimizer([muon, fallback], [muon_hidden_matrix_filter])
    if callable(learning_rate):
        raise OptimizerContractFault(
            "schedule_free_candidate_owns_its_schedule_and_requires_scalar_learning_rate"
        )
    return ScheduleFreeAdamW(
        learning_rate=float(learning_rate),
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        mx=mx,
        optim=optim,
    )


def AdEMAMix(
    *,
    learning_rate: Any,
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    beta3_warmup_steps: int,
    alpha_warmup_steps: int,
    eps: float,
    weight_decay: float,
    optim: Any,
    mx: Any,
) -> Any:
    """Create Apple's reference fast/slow-EMA AdEMAMix update for MLX."""

    from mlx.utils import tree_map

    class _AdEMAMix(optim.Optimizer):
        def __init__(self) -> None:
            super().__init__()
            self._maybe_schedule("learning_rate", learning_rate)
            self.beta1 = float(beta1)
            self.beta2 = float(beta2)
            self.beta3_final = float(beta3)
            self.alpha_final = float(alpha)
            self.beta3_warmup_steps = int(beta3_warmup_steps)
            self.alpha_warmup_steps = int(alpha_warmup_steps)
            self.eps = float(eps)
            self.weight_decay = float(weight_decay)
            self.theseus_policy_id = "ademamix_fast_slow_ema_v1"
            self.state["effective_alpha"] = mx.array(0.0, dtype=mx.float32)
            self.state["effective_beta3"] = mx.array(
                self.beta1, dtype=mx.float32
            )

        def init_single(self, parameter: Any, state: dict[str, Any]) -> None:
            if self.beta1:
                state["exp_avg_fast"] = mx.zeros_like(parameter)
            state["exp_avg_slow"] = mx.zeros_like(parameter)
            state["exp_avg_sq"] = mx.zeros_like(parameter)

        def apply_gradients(self, gradients: dict, parameters: dict) -> dict:
            if not self._initialized:
                self.init(gradients)
            for name, scheduler in self._schedulers.items():
                self.state[name] = scheduler(self.step)
            step = self.step + 1
            step_fp32 = step.astype(mx.float32)
            alpha_fraction = (
                mx.minimum(
                    step_fp32 / float(self.alpha_warmup_steps),
                    mx.array(1.0, dtype=mx.float32),
                )
                if self.alpha_warmup_steps
                else mx.array(1.0, dtype=mx.float32)
            )
            alpha_now = self.alpha_final * alpha_fraction
            if self.beta3_warmup_steps:
                beta3_fraction = mx.minimum(
                    step_fp32 / float(self.beta3_warmup_steps),
                    mx.array(1.0, dtype=mx.float32),
                )
                start_half_life = (
                    math.log(0.5) / math.log(self.beta1 + 1e-8) - 1.0
                )
                end_half_life = (
                    math.log(0.5) / math.log(self.beta3_final + 1e-8) - 1.0
                )
                half_life = (
                    (1.0 - beta3_fraction) * start_half_life
                    + beta3_fraction * end_half_life
                )
                beta3_now = mx.power(
                    mx.array(0.5, dtype=mx.float32),
                    1.0 / (half_life + 1.0),
                )
            else:
                beta3_now = mx.array(self.beta3_final, dtype=mx.float32)
            self.state.update(
                {
                    "step": step,
                    "effective_alpha": alpha_now,
                    "effective_beta3": beta3_now,
                }
            )
            return tree_map(self.apply_single, gradients, parameters, self.state)

        def apply_single(
            self, gradient: Any, parameter: Any, state: dict[str, Any]
        ) -> Any:
            gradient = gradient.astype(parameter.dtype)
            if self.beta1:
                fast = self.beta1 * state["exp_avg_fast"] + (
                    1.0 - self.beta1
                ) * gradient
                state["exp_avg_fast"] = fast
            else:
                fast = gradient
            beta3_now = self.state["effective_beta3"].astype(gradient.dtype)
            slow = beta3_now * state["exp_avg_slow"] + (
                1.0 - beta3_now
            ) * gradient
            square = self.beta2 * state["exp_avg_sq"] + (
                1.0 - self.beta2
            ) * mx.square(gradient)
            state["exp_avg_slow"] = slow
            state["exp_avg_sq"] = square

            step = self.step.astype(gradient.dtype)
            correction1 = 1.0 - self.beta1**step
            correction2 = 1.0 - self.beta2**step
            denominator = mx.sqrt(square) / mx.sqrt(correction2) + self.eps
            update = (
                fast / correction1
                + self.state["effective_alpha"].astype(gradient.dtype) * slow
            ) / denominator
            if self.weight_decay:
                update = update + self.weight_decay * parameter
            return parameter - self.learning_rate.astype(gradient.dtype) * update

    return _AdEMAMix()


def AdamMini(
    *,
    learning_rate: Any,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    optim: Any,
    mx: Any,
) -> Any:
    """Create Adam-mini v1.1 with Theseus-specific content-bound partitions."""

    from mlx.utils import tree_flatten, tree_merge

    class _AdamMiniPartition(optim.Optimizer):
        def __init__(self, mode: str, local_weight_decay: float) -> None:
            super().__init__()
            self._maybe_schedule("learning_rate", learning_rate)
            self.mode = mode
            self.beta1 = float(beta1)
            self.beta2 = float(beta2)
            self.eps = float(eps)
            self.weight_decay = float(local_weight_decay)
            self.head_numel = int(dim) * int(dim) // int(num_heads)

        def init_single(self, parameter: Any, state: dict[str, Any]) -> None:
            state["m"] = mx.zeros_like(parameter)
            if self.mode == "adam":
                state["v"] = mx.zeros_like(parameter)
            elif self.mode == "head":
                if int(parameter.size) % self.head_numel:
                    raise OptimizerContractFault(
                        "adam_mini_query_key_head_partition_invalid"
                    )
                state["vmean"] = mx.zeros(
                    (int(parameter.size) // self.head_numel, 1),
                    dtype=parameter.dtype,
                )
            elif self.mode == "row":
                if int(parameter.ndim) != 2:
                    raise OptimizerContractFault(
                        "adam_mini_row_partition_requires_matrix"
                    )
                state["vmean"] = mx.zeros(
                    (int(parameter.shape[0]), 1), dtype=parameter.dtype
                )
            else:
                state["vmean"] = mx.array(0.0, dtype=parameter.dtype)

        def apply_single(
            self, gradient: Any, parameter: Any, state: dict[str, Any]
        ) -> Any:
            step = self.step.astype(gradient.dtype)
            moment = self.beta1 * state["m"] + (
                1.0 - self.beta1
            ) * gradient
            state["m"] = moment
            if self.mode == "adam":
                square = self.beta2 * state["v"] + (
                    1.0 - self.beta2
                ) * mx.square(gradient)
                state["v"] = square
                variance = square
            elif self.mode == "head":
                reshaped = gradient.reshape(-1, self.head_numel)
                current = mx.mean(mx.square(reshaped), axis=1, keepdims=True)
                variance = self.beta2 * state["vmean"] + (
                    1.0 - self.beta2
                ) * current
                state["vmean"] = variance
                variance = mx.broadcast_to(
                    variance, reshaped.shape
                ).reshape(gradient.shape)
            elif self.mode == "row":
                current = mx.mean(mx.square(gradient), axis=1, keepdims=True)
                variance = self.beta2 * state["vmean"] + (
                    1.0 - self.beta2
                ) * current
                state["vmean"] = variance
                variance = mx.broadcast_to(variance, gradient.shape)
            else:
                current = mx.mean(mx.square(gradient))
                variance = self.beta2 * state["vmean"] + (
                    1.0 - self.beta2
                ) * current
                state["vmean"] = variance

            correction1 = 1.0 - self.beta1**step
            correction2 = 1.0 - self.beta2**step
            denominator = mx.sqrt(variance) / mx.sqrt(correction2) + self.eps
            decayed = parameter * (
                1.0
                - self.learning_rate.astype(parameter.dtype)
                * self.weight_decay
            )
            return decayed - self.learning_rate.astype(gradient.dtype) * (
                moment / correction1
            ) / denominator

    class _AdamMiniMultiOptimizer(optim.MultiOptimizer):
        """MLX MultiOptimizer variant that safely skips empty partitions."""

        def init(self, parameters: dict) -> None:
            for child, partition in zip(
                self.optimizers, self._split_dictionary(parameters)
            ):
                if tree_flatten(partition):
                    child.init(partition)

        def apply_gradients(self, gradients: dict, parameters: dict) -> dict:
            updated: dict[str, Any] = {}
            for child, partition in zip(
                self.optimizers, self._split_dictionary(gradients)
            ):
                if tree_flatten(partition):
                    updated = tree_merge(
                        updated,
                        child.apply_gradients(partition, parameters),
                    )
            return updated

    children = [
        _AdamMiniPartition("adam", 0.0),
        _AdamMiniPartition("head", weight_decay),
        _AdamMiniPartition("row", weight_decay),
        _AdamMiniPartition("block", 0.0),
        _AdamMiniPartition("block", weight_decay),
    ]
    partitions = ["adam", "head", "row", "block_no_decay"]
    optimizer = _AdamMiniMultiOptimizer(
        children,
        [
            lambda path, value, expected=expected: adam_mini_partition(
                path, value
            )
            == expected
            for expected in partitions
        ],
    )
    optimizer.theseus_policy_id = "adam_mini_hessian_partition_v1_1"
    optimizer.adam_mini_dim = int(dim)
    optimizer.adam_mini_num_heads = int(num_heads)
    optimizer.adam_mini_num_kv_heads = int(num_kv_heads)
    optimizer.adam_mini_head_numel = int(dim) * int(dim) // int(num_heads)
    optimizer.adam_mini_partition_policy = {
        "adam": sorted({"bias"}),
        "head": sorted(ADAM_MINI_QUERY_KEY_NAMES),
        "row": sorted(
            ADAM_MINI_EMBEDDING_NAMES
            | ADAM_MINI_OUTPUT_NAMES
            | ADAM_MINI_VALUE_NAMES
            | ADAM_MINI_ATTENTION_OUTPUT_NAMES
            | ADAM_MINI_MLP_NAMES
        ),
        "block_no_decay": sorted({"norm", ".ln"}),
        "block": ["fallback"],
    }
    return optimizer


def AdamWWithBFloat16Moments(
    *,
    learning_rate: Any,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    bias_correction: bool,
    optim: Any,
    mx: Any,
) -> Any:
    """AdamW with FP32 update math and compact persistent moment custody.

    Project Theseus keeps an authoritative FP32 parameter model for this route.
    The first and second moments are the only large state made bfloat16; every
    update expands them to FP32, performs the AdamW equations in FP32, then
    explicitly rounds the new persistent state back to bfloat16.  This avoids
    carrying two additional full-size FP32 model copies into the next backward
    pass while retaining the exponent range needed by the squared moment.
    """

    from mlx.utils import tree_map_with_path

    class _AdamWWithBFloat16Moments(optim.AdamW):
        def __init__(self) -> None:
            super().__init__(
                learning_rate=learning_rate,
                betas=[float(beta1), float(beta2)],
                eps=float(eps),
                weight_decay=float(weight_decay),
                bias_correction=bool(bias_correction),
            )
            self.persistent_moment_dtype = "bfloat16"
            self.update_math_dtype = "float32"
            self.transactional_parameter_updates = True

        def init_single(self, parameter: Any, state: dict[str, Any]) -> None:
            state["m"] = mx.zeros(parameter.shape, dtype=mx.bfloat16)
            state["v"] = mx.zeros(parameter.shape, dtype=mx.bfloat16)

        def apply_single(
            self, gradient: Any, parameter: Any, state: dict[str, Any]
        ) -> Any:
            gradient_fp32 = gradient.astype(mx.float32)
            parameter_fp32 = parameter.astype(mx.float32)
            moment = state["m"].astype(mx.float32)
            square_moment = state["v"].astype(mx.float32)
            moment = self.betas[0] * moment + (1.0 - self.betas[0]) * gradient_fp32
            square_moment = self.betas[1] * square_moment + (
                1.0 - self.betas[1]
            ) * mx.square(gradient_fp32)
            state["m"] = moment.astype(mx.bfloat16)
            state["v"] = square_moment.astype(mx.bfloat16)

            learning_rate_fp32 = self.learning_rate.astype(mx.float32)
            decayed = parameter_fp32 * (
                1.0 - learning_rate_fp32 * self.weight_decay
            )
            if self.bias_correction:
                step = self.step.astype(mx.float32)
                corrected_rate = learning_rate_fp32 / (
                    1.0 - self.betas[0] ** step
                )
                corrected_scale = mx.rsqrt(1.0 - self.betas[1] ** step)
                updated = decayed - corrected_rate * moment / (
                    mx.sqrt(square_moment) * corrected_scale + self.eps
                )
            else:
                updated = decayed - learning_rate_fp32 * moment / (
                    mx.sqrt(square_moment) + self.eps
                )
            return updated.astype(parameter.dtype)

        def update(self, model: Any, gradients: dict[str, Any]) -> None:
            """Commit one parameter at a time to bound lazy update-graph memory."""

            if not self._initialized:
                self.init(gradients)
            for name, scheduler in self._schedulers.items():
                self.state[name] = scheduler(self.step)
            self.state["step"] = self.step + 1
            parameters = model.trainable_parameters()

            def apply_and_commit(
                path: str,
                gradient: Any,
                parameter: Any,
                state: dict[str, Any],
            ) -> Any:
                updated = self.apply_single(gradient, parameter, state)
                mx.eval(updated, state["m"], state["v"])
                model.load_weights([(path, updated)], strict=False)
                return updated

            # The returned tree shares the committed arrays with the model and
            # is intentionally not retained.  Each lazy FP32 expansion is fully
            # evaluated before the next tensor's graph is constructed.
            tree_map_with_path(
                apply_and_commit,
                gradients,
                parameters,
                self.state,
            )

    return _AdamWWithBFloat16Moments()


def ScheduleFreeAdamW(
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    warmup_steps: int,
    mx: Any,
    optim: Any,
) -> Any:
    """Create the reference x/y/z schedule-free AdamW update for MLX."""

    from mlx.utils import tree_map

    class _ScheduleFreeAdamW(optim.Optimizer):
        def __init__(self) -> None:
            super().__init__()
            self.state.update(
                {
                    "learning_rate": mx.array(float(learning_rate)),
                    "weight_sum": mx.array(0.0, dtype=mx.float32),
                    "lr_max": mx.array(0.0, dtype=mx.float32),
                    "scheduled_lr": mx.array(0.0, dtype=mx.float32),
                    "ckp1": mx.array(0.0, dtype=mx.float32),
                }
            )
            self.beta1 = float(beta1)
            self.beta2 = float(beta2)
            self.eps = float(eps)
            self.weight_decay = float(weight_decay)
            self.warmup_steps = int(warmup_steps)
            self.weight_lr_power = 2.0

        def init_single(self, parameter: Any, state: dict[str, Any]) -> None:
            state["z"] = mx.array(parameter)
            state["x"] = mx.array(parameter)
            state["y"] = mx.array(parameter)
            state["exp_avg_sq"] = mx.zeros_like(parameter)

        def apply_gradients(self, gradients: dict, parameters: dict) -> dict:
            if not self._initialized:
                self.init(gradients)
            step = self.step + 1
            warmup = (
                mx.minimum(step.astype(mx.float32) / float(self.warmup_steps), 1.0)
                if self.warmup_steps
                else mx.array(1.0, dtype=mx.float32)
            )
            scheduled_lr = self.learning_rate * warmup
            lr_max = mx.maximum(self.state["lr_max"], scheduled_lr)
            weight = lr_max**self.weight_lr_power
            weight_sum = self.state["weight_sum"] + weight
            ckp1 = mx.where(weight_sum > 0.0, weight / weight_sum, 0.0)
            self.state.update(
                {
                    "step": step,
                    "scheduled_lr": scheduled_lr,
                    "lr_max": lr_max,
                    "weight_sum": weight_sum,
                    "ckp1": ckp1,
                }
            )
            return tree_map(self.apply_single, gradients, parameters, self.state)

        def apply_single(
            self, gradient: Any, parameter: Any, state: dict[str, Any]
        ) -> Any:
            del parameter
            step = self.step.astype(gradient.dtype)
            lr = self.state["scheduled_lr"].astype(gradient.dtype)
            exp_avg_sq = self.beta2 * state["exp_avg_sq"] + (
                1.0 - self.beta2
            ) * mx.square(gradient)
            bias_correction = 1.0 - self.beta2**step
            update = gradient / (mx.sqrt(exp_avg_sq / bias_correction) + self.eps)
            z = state["z"]
            y = state["y"]
            if self.weight_decay:
                z = z - lr * self.weight_decay * y
            z = z - lr * update
            x = (1.0 - self.state["ckp1"]) * state["x"] + self.state["ckp1"] * z
            y = self.beta1 * x + (1.0 - self.beta1) * z
            state.update({"z": z, "x": x, "y": y, "exp_avg_sq": exp_avg_sq})
            return y

        def evaluation_parameters(self, parameters: dict) -> dict:
            if not self._initialized:
                self.init(parameters)
            return tree_map(lambda _parameter, state: state["x"], parameters, self.state)

        def training_parameters(self, parameters: dict) -> dict:
            if not self._initialized:
                self.init(parameters)
            return tree_map(lambda _parameter, state: state["y"], parameters, self.state)

        def set_evaluation_iterate(self, model: Any) -> None:
            model.update(self.evaluation_parameters(model.trainable_parameters()))

        def set_training_iterate(self, model: Any) -> None:
            model.update(self.training_parameters(model.trainable_parameters()))

    return _ScheduleFreeAdamW()


def optimizer_state_kind(optimizer: Any) -> str:
    if (
        getattr(optimizer, "theseus_policy_id", "")
        == "ademamix_fast_slow_ema_v1"
    ):
        return "ademamix_fast_slow_and_squared_ema"
    if (
        getattr(optimizer, "theseus_policy_id", "")
        == "adam_mini_hessian_partition_v1_1"
    ):
        return "adam_mini_full_first_moment_partitioned_second_moment"
    if (
        getattr(optimizer, "theseus_policy_id", "")
        == "adafactor_factored_second_moment_v1"
    ):
        return "adafactor_factored_matrices_unfactored_vectors_scalars"
    if getattr(optimizer, "persistent_moment_dtype", "") == "bfloat16":
        return "adamw_bfloat16_moments_fp32_transactional_update_math"
    if hasattr(optimizer, "set_evaluation_iterate"):
        return "schedule_free_x_y_z"
    if optimizer.__class__.__name__ == "MultiOptimizer":
        return "muon_hidden_matrices_plus_adamw_fallback"
    return "adamw_moments"
