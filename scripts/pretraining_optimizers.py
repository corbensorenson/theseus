#!/usr/bin/env python3
"""Canonical MLX optimizer implementations for pre-training candidates."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class OptimizerContractFault(ValueError):
    pass


OPTIMIZER_IDS = {
    "adafactor_mlx",
    "adamw_mlx",
    "adamw_bfloat16_moments_mlx",
    "muon_mlx",
    "schedule_free_adamw_mlx",
}


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
) -> Any:
    if optimizer_id not in OPTIMIZER_IDS:
        raise OptimizerContractFault(f"optimizer_unknown:{optimizer_id}")
    if weight_decay < 0.0 or eps <= 0.0:
        raise OptimizerContractFault("optimizer_numeric_contract_invalid")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise OptimizerContractFault("optimizer_beta_contract_invalid")
    if warmup_steps < 0:
        raise OptimizerContractFault("optimizer_warmup_invalid")
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
