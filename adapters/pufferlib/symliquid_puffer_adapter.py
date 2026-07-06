"""Optional PufferLib bridge for SymLiquid policies.

This module intentionally does not depend on PufferLib at import time. It gives
us a stable adapter surface that can be wired into PufferLib/Ocean environments
after the local Python environment has PufferLib installed.

The adapter is for environment interaction only. It must not call external model
providers or use third-party model inference.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

_RUST_FFI_LIB: Any | None = None
_RUST_FFI_LIB_PATH: Path | None = None


@dataclasses.dataclass
class SymLiquidPolicyArtifact:
    labels: list[str]
    hv_dim: int
    output_dim: int
    weights: list[float]
    bias: list[float]
    feature_set: str = "numeric_hash_vsa"

    @classmethod
    def load(cls, path: str | Path) -> "SymLiquidPolicyArtifact":
        data = json.loads(Path(path).read_text())
        return cls(
            labels=list(data["labels"]),
            hv_dim=int(data["hv_dim"]),
            output_dim=int(data["output_dim"]),
            weights=[float(x) for x in data["weights"]],
            bias=[float(x) for x in data["bias"]],
            feature_set=str(data.get("feature_set", "numeric_hash_vsa")),
        )


def memory_stride_for_feature_set(feature_set: str) -> int:
    if feature_set == "slot_tmaze_recurrent_linear_v1":
        return 2
    return 1


class SymLiquidPufferPolicy:
    """Small local policy adapter for discrete Puffer-style action batches.

    The current version maps numeric observations into deterministic hashed
    hypervector features and applies a local linear readout artifact. It is a
    bridge surface, not the final high-performance CUDA rollout path.
    """

    def __init__(self, artifact: SymLiquidPolicyArtifact):
        if len(artifact.weights) != artifact.output_dim * artifact.hv_dim:
            raise ValueError("artifact weights do not match output_dim * hv_dim")
        if len(artifact.bias) != artifact.output_dim:
            raise ValueError("artifact bias does not match output_dim")
        self.artifact = artifact
        try:
            import numpy as np

            self._np = np
            self._weights_np = np.asarray(artifact.weights, dtype=np.float32).reshape(
                artifact.output_dim, artifact.hv_dim
            )
            self._bias_np = np.asarray(artifact.bias, dtype=np.float32)
        except Exception:  # pragma: no cover - optional speed path
            self._np = None
            self._weights_np = None
            self._bias_np = None
        self._memory_state: list[float] = []

    def act(self, observations: Iterable[Any]) -> list[int]:
        observations = list(observations)
        stride = memory_stride_for_feature_set(self.artifact.feature_set)
        required = len(observations) * stride
        if len(self._memory_state) < required:
            self._memory_state.extend([0.0] * (required - len(self._memory_state)))
        return [self._act_one(obs, env_idx) for env_idx, obs in enumerate(observations)]

    def _act_one(self, observation: Any, env_idx: int = 0) -> int:
        sparse, norm = artifact_sparse_features(
            observation, self.artifact, self._memory_state, env_idx
        )
        if self._np is not None:
            logits = self._bias_np.copy()
            for idx, value in sparse:
                logits += self._weights_np[:, idx] * (value / norm)
            return int(self._np.argmax(logits))

        logits = list(self.artifact.bias)
        for idx, value in sparse:
            scaled = value / norm
            for out_idx in range(self.artifact.output_dim):
                logits[out_idx] += (
                    self.artifact.weights[out_idx * self.artifact.hv_dim + idx] * scaled
                )
        return max(range(len(logits)), key=logits.__getitem__)


class RustFfiPolicy:
    """ctypes bridge to the Rust policy scorer.

    This keeps recurrent policy/state math behind the Rust ABI while Python
    remains responsible for environment orchestration during adapter tests.
    """

    def __init__(self, artifact_path: str | Path, num_envs: int):
        self.artifact_path = Path(artifact_path)
        self.num_envs = int(num_envs)
        self._lib_path, self._lib = load_rust_ffi_library()
        self._bind_library()
        self._handle = self._lib.symliquid_policy_load(
            str(self.artifact_path).encode("utf-8"), ctypes.c_size_t(self.num_envs)
        )
        if not self._handle:
            raise RuntimeError(f"failed to load SymLiquid Rust FFI policy artifact: {self.artifact_path}")

    @classmethod
    def from_params(
        cls,
        feature_set: str,
        hv_dim: int,
        output_dim: int,
        weights: list[float],
        bias: list[float],
        num_envs: int,
    ) -> "RustFfiPolicy":
        policy = cls.__new__(cls)
        policy.artifact_path = None
        policy.num_envs = int(num_envs)
        policy._lib_path, policy._lib = load_rust_ffi_library()
        policy._bind_library()
        weights_array = (ctypes.c_float * len(weights))(*weights)
        bias_array = (ctypes.c_float * len(bias))(*bias)
        policy._handle = policy._lib.symliquid_policy_from_parts(
            feature_set.encode("utf-8"),
            ctypes.c_size_t(hv_dim),
            ctypes.c_size_t(output_dim),
            weights_array,
            ctypes.c_size_t(len(weights)),
            bias_array,
            ctypes.c_size_t(len(bias)),
            ctypes.c_size_t(policy.num_envs),
        )
        if not policy._handle:
            raise RuntimeError("failed to construct SymLiquid Rust FFI policy from parameters")
        return policy

    def _bind_library(self) -> None:
        self._lib.symliquid_policy_load.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self._lib.symliquid_policy_load.restype = ctypes.c_void_p
        self._lib.symliquid_policy_from_parts.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._lib.symliquid_policy_from_parts.restype = ctypes.c_void_p
        self._lib.symliquid_policy_free.argtypes = [ctypes.c_void_p]
        self._lib.symliquid_policy_free.restype = None
        self._lib.symliquid_policy_reset.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._lib.symliquid_policy_reset.restype = ctypes.c_int
        self._lib.symliquid_policy_output_dim.argtypes = [ctypes.c_void_p]
        self._lib.symliquid_policy_output_dim.restype = ctypes.c_size_t
        self._lib.symliquid_policy_act.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.symliquid_policy_act.restype = ctypes.c_int

    @property
    def library_path(self) -> str:
        return str(self._lib_path)

    def reset(self, num_envs: int | None = None) -> None:
        num_envs = self.num_envs if num_envs is None else int(num_envs)
        status = self._lib.symliquid_policy_reset(self._handle, ctypes.c_size_t(num_envs))
        if status != 0:
            raise RuntimeError(f"SymLiquid Rust FFI policy_reset failed with status {status}")
        self.num_envs = num_envs

    def act(self, observations: Iterable[Any]) -> list[int]:
        rows = [flatten_numeric(observation) for observation in observations]
        if not rows:
            return []
        obs_dim = max(1, max(len(row) for row in rows))
        flat_values: list[float] = []
        for row in rows:
            flat_values.extend(row)
            if len(row) < obs_dim:
                flat_values.extend([0.0] * (obs_dim - len(row)))
        obs_array = (ctypes.c_float * len(flat_values))(*flat_values)
        actions = (ctypes.c_int * len(rows))()
        status = self._lib.symliquid_policy_act(
            self._handle,
            obs_array,
            ctypes.c_size_t(len(rows)),
            ctypes.c_size_t(obs_dim),
            actions,
        )
        if status != 0:
            raise RuntimeError(f"SymLiquid Rust FFI policy_act failed with status {status}")
        return [int(action) for action in actions]

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.symliquid_policy_free(self._handle)
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup path
        try:
            self.close()
        except Exception:
            pass


def find_rust_ffi_library() -> Path:
    env_path = os.environ.get("SYMLIQUID_FFI_DLL")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    root = Path(__file__).resolve().parents[2]
    if sys.platform == "win32":
        filename = "symliquid_ffi.dll"
    elif sys.platform == "darwin":
        filename = "libsymliquid_ffi.dylib"
    else:
        filename = "libsymliquid_ffi.so"
    candidates.extend(
        [
            root / "target" / "release" / filename,
            root / "target" / "debug" / filename,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "SymLiquid Rust FFI library not found. Build it with "
        f"`cargo build --release -p symliquid-ffi` or set SYMLIQUID_FFI_DLL. Searched: {searched}"
    )


def load_rust_ffi_library() -> tuple[Path, Any]:
    global _RUST_FFI_LIB, _RUST_FFI_LIB_PATH
    if _RUST_FFI_LIB is None or _RUST_FFI_LIB_PATH is None:
        _RUST_FFI_LIB_PATH = find_rust_ffi_library()
        _RUST_FFI_LIB = ctypes.CDLL(str(_RUST_FFI_LIB_PATH))
    return _RUST_FFI_LIB_PATH, _RUST_FFI_LIB


class PufferStyleToyEnv:
    """Tiny vectorized environment with a Puffer-like reset/step surface."""

    def __init__(self, num_envs: int, obs_dim: int, action_modulo: int = 4):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_modulo = max(1, action_modulo)
        self.ticks = [0 for _ in range(num_envs)]

    def reset(self) -> tuple[list[list[float]], list[dict[str, Any]]]:
        self.ticks = [0 for _ in range(self.num_envs)]
        return self._observations(), [{"reset": True} for _ in range(self.num_envs)]

    def step(
        self, actions: Iterable[int]
    ) -> tuple[list[list[float]], list[float], list[bool], list[bool], list[dict[str, Any]]]:
        rewards: list[float] = []
        terminals: list[bool] = []
        truncations: list[bool] = []
        infos: list[dict[str, Any]] = []
        for env_idx, action in enumerate(actions):
            target = self._target(env_idx)
            reward = 1.0 if int(action) % self.action_modulo == target else 0.0
            rewards.append(reward)
            self.ticks[env_idx] += 1
            terminals.append(self.ticks[env_idx] % 32 == 0)
            truncations.append(False)
            infos.append({"target": target, "action": int(action)})
            if terminals[-1]:
                self.ticks[env_idx] = 0
        return self._observations(), rewards, terminals, truncations, infos

    def _observations(self) -> list[list[float]]:
        observations = []
        for env_idx, tick in enumerate(self.ticks):
            row = []
            for obs_idx in range(self.obs_dim):
                value = ((env_idx * 17 + tick * 7 + obs_idx * 3) % 23) / 11.0 - 1.0
                row.append(value)
            observations.append(row)
        return observations

    def _target(self, env_idx: int) -> int:
        return int((env_idx + self.ticks[env_idx]) % self.action_modulo)


class OceanCartpoleEnv:
    """Vectorized local port of PufferLib Ocean CartPole dynamics.

    This keeps the adapter on a real RL control surface even when the vendored
    PufferLib C extension is not built on the local machine.
    """

    def __init__(self, num_envs: int, seed: int = 0):
        self.num_envs = num_envs
        self.rng = random.Random(seed)
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.gravity = 9.8
        self.force_mag = 10.0
        self.tau = 0.02
        self.max_steps = 200
        self.x_threshold = 2.4
        self.theta_threshold = 12 * 2 * math.pi / 360
        self.states = [[0.0, 0.0, 0.0, 0.0] for _ in range(num_envs)]
        self.ticks = [0 for _ in range(num_envs)]
        self.episode_returns = [0.0 for _ in range(num_envs)]

    def reset(self) -> tuple[list[list[float]], list[dict[str, Any]]]:
        for env_idx in range(self.num_envs):
            self._reset_one(env_idx)
        return self._observations(), [{"env": "ocean_cartpole", "reset": True} for _ in range(self.num_envs)]

    def step(
        self, actions: Iterable[int]
    ) -> tuple[list[list[float]], list[float], list[bool], list[bool], list[dict[str, Any]]]:
        rewards: list[float] = []
        terminals: list[bool] = []
        truncations: list[bool] = []
        infos: list[dict[str, Any]] = []

        for env_idx, action in enumerate(actions):
            x, x_dot, theta, theta_dot = self.states[env_idx]
            force = self.force_mag if int(action) % 2 else -self.force_mag
            costheta = math.cos(theta)
            sintheta = math.sin(theta)
            total_mass = self.cart_mass + self.pole_mass
            polemass_length = total_mass + self.pole_mass
            temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.pole_length * (4.0 / 3.0 - total_mass * costheta * costheta / total_mass)
            )
            xacc = temp - polemass_length * thetaacc * costheta / total_mass

            x += self.tau * x_dot
            x_dot += self.tau * xacc
            theta += self.tau * theta_dot
            theta_dot += self.tau * thetaacc
            self.ticks[env_idx] += 1

            terminal = (
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold
                or theta > self.theta_threshold
            )
            truncation = self.ticks[env_idx] >= self.max_steps
            done = terminal or truncation
            reward = 0.0 if done else 1.0
            self.episode_returns[env_idx] += reward
            self.states[env_idx] = [x, x_dot, theta, theta_dot]
            info = {
                "env": "ocean_cartpole",
                "action": int(action) % 2,
                "episode_return": self.episode_returns[env_idx],
                "tick": self.ticks[env_idx],
            }
            if done:
                info["final_observation"] = list(self.states[env_idx])
                self._reset_one(env_idx)

            rewards.append(reward)
            terminals.append(terminal)
            truncations.append(truncation)
            infos.append(info)

        return self._observations(), rewards, terminals, truncations, infos

    def _reset_one(self, env_idx: int) -> None:
        self.states[env_idx] = [self.rng.random() * 0.08 - 0.04 for _ in range(4)]
        self.ticks[env_idx] = 0
        self.episode_returns[env_idx] = 0.0

    def _observations(self) -> list[list[float]]:
        return [list(state) for state in self.states]


class OceanChainMdpEnv:
    """Vectorized local port of PufferLib Ocean Chain MDP."""

    def __init__(self, num_envs: int, size: int = 16, seed: int = 0):
        self.num_envs = num_envs
        self.size = max(4, size)
        self.rng = random.Random(seed)
        self.states = [1 for _ in range(num_envs)]
        self.ticks = [0 for _ in range(num_envs)]

    def reset(self) -> tuple[list[list[float]], list[dict[str, Any]]]:
        self.states = [1 for _ in range(self.num_envs)]
        self.ticks = [0 for _ in range(self.num_envs)]
        return self._observations(), [{"env": "ocean_chain_mdp", "reset": True} for _ in range(self.num_envs)]

    def step(
        self, actions: Iterable[int]
    ) -> tuple[list[list[float]], list[float], list[bool], list[bool], list[dict[str, Any]]]:
        rewards: list[float] = []
        terminals: list[bool] = []
        truncations: list[bool] = []
        infos: list[dict[str, Any]] = []
        for env_idx, action in enumerate(actions):
            self.ticks[env_idx] += 1
            delta = 1 if int(action) % 2 else -1
            self.states[env_idx] = max(0, min(self.size - 1, self.states[env_idx] + delta))
            reward = 0.0
            if self.states[env_idx] == 0:
                reward = 0.001
            elif self.states[env_idx] == self.size - 1:
                reward = 1.0
            terminal = self.ticks[env_idx] == self.size + 9
            if terminal:
                self.states[env_idx] = 1
                self.ticks[env_idx] = 0
            rewards.append(reward)
            terminals.append(terminal)
            truncations.append(False)
            infos.append({"env": "ocean_chain_mdp", "state": self.states[env_idx]})
        return self._observations(), rewards, terminals, truncations, infos

    def _observations(self) -> list[list[float]]:
        scale = max(1, self.size - 1)
        return [[state / scale] for state in self.states]


class OceanMemoryEnv:
    """Vectorized local port of PufferLib Ocean Memory."""

    def __init__(self, num_envs: int, length: int = 8, seed: int = 0):
        self.num_envs = num_envs
        self.length = max(2, length)
        self.rng = random.Random(seed)
        self.goals = [1 for _ in range(num_envs)]
        self.ticks = [0 for _ in range(num_envs)]
        self.visible = [True for _ in range(num_envs)]

    def reset(self) -> tuple[list[list[float]], list[dict[str, Any]]]:
        for env_idx in range(self.num_envs):
            self._reset_one(env_idx)
        return self._observations(), [{"env": "ocean_memory", "reset": True} for _ in range(self.num_envs)]

    def step(
        self, actions: Iterable[int]
    ) -> tuple[list[list[float]], list[float], list[bool], list[bool], list[dict[str, Any]]]:
        rewards: list[float] = []
        terminals: list[bool] = []
        truncations: list[bool] = []
        infos: list[dict[str, Any]] = []
        for env_idx, action in enumerate(actions):
            self.visible[env_idx] = False
            self.ticks[env_idx] += 1
            terminal = self.ticks[env_idx] >= self.length
            reward = 0.0
            if terminal:
                correct = (int(action) % 2 == 0 and self.goals[env_idx] == -1) or (
                    int(action) % 2 == 1 and self.goals[env_idx] == 1
                )
                reward = 1.0 if correct else 0.0
                self._reset_one(env_idx)
            rewards.append(reward)
            terminals.append(terminal)
            truncations.append(False)
            infos.append({"env": "ocean_memory", "goal": self.goals[env_idx]})
        return self._observations(), rewards, terminals, truncations, infos

    def _reset_one(self, env_idx: int) -> None:
        self.goals[env_idx] = -1 if self.rng.randrange(2) == 0 else 1
        self.ticks[env_idx] = 0
        self.visible[env_idx] = True

    def _observations(self) -> list[list[float]]:
        return [[float(goal if visible else 0.0)] for goal, visible in zip(self.goals, self.visible)]


class OceanNoisyMemoryEnv:
    """Delayed noisy-evidence memory task for recurrent state pressure tests.

    Each episode emits several noisy binary evidence samples for a hidden goal,
    then a blank delay, then a final decision flag. The policy must accumulate
    evidence over time; storing only the last cue is deliberately suboptimal.
    """

    def __init__(
        self,
        num_envs: int,
        length: int = 12,
        evidence_steps: int = 5,
        evidence_accuracy: float = 0.75,
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.length = max(4, length)
        self.evidence_steps = max(1, min(evidence_steps, self.length - 2))
        self.evidence_accuracy = min(1.0, max(0.5, evidence_accuracy))
        self.rng = random.Random(seed)
        self.goals = [1 for _ in range(num_envs)]
        self.ticks = [0 for _ in range(num_envs)]
        self.evidence = [[0.0 for _ in range(self.evidence_steps)] for _ in range(num_envs)]
        self.distractors = [[0.0 for _ in range(self.length)] for _ in range(num_envs)]

    def reset(self) -> tuple[list[list[float]], list[dict[str, Any]]]:
        for env_idx in range(self.num_envs):
            self._reset_one(env_idx)
        return self._observations(), [{"env": "ocean_noisy_memory", "reset": True} for _ in range(self.num_envs)]

    def step(
        self, actions: Iterable[int]
    ) -> tuple[list[list[float]], list[float], list[bool], list[bool], list[dict[str, Any]]]:
        rewards: list[float] = []
        terminals: list[bool] = []
        truncations: list[bool] = []
        infos: list[dict[str, Any]] = []
        for env_idx, action in enumerate(actions):
            decision_tick = self.ticks[env_idx] >= self.length - 1
            reward = 0.0
            terminal = False
            if decision_tick:
                correct_action = 1 if self.goals[env_idx] > 0 else 0
                reward = 1.0 if int(action) % 2 == correct_action else 0.0
                terminal = True
                goal = self.goals[env_idx]
                self._reset_one(env_idx)
            else:
                goal = self.goals[env_idx]
                self.ticks[env_idx] += 1
            rewards.append(reward)
            terminals.append(terminal)
            truncations.append(False)
            infos.append({"env": "ocean_noisy_memory", "goal": goal})
        return self._observations(), rewards, terminals, truncations, infos

    def _reset_one(self, env_idx: int) -> None:
        goal = -1 if self.rng.randrange(2) == 0 else 1
        self.goals[env_idx] = goal
        self.ticks[env_idx] = 0
        self.evidence[env_idx] = [
            float(goal if self.rng.random() < self.evidence_accuracy else -goal)
            for _ in range(self.evidence_steps)
        ]
        self.distractors[env_idx] = [
            float(-1 if self.rng.randrange(2) == 0 else 1) for _ in range(self.length)
        ]

    def _observations(self) -> list[list[float]]:
        observations = []
        for env_idx, tick in enumerate(self.ticks):
            reset_phase = 1.0 if tick == 0 else 0.0
            time_fraction = tick / max(1, self.length - 1)
            if tick < self.evidence_steps:
                observations.append(
                    [self.evidence[env_idx][tick], 1.0, 0.0, time_fraction, reset_phase]
                )
            elif tick >= self.length - 1:
                observations.append([0.0, 0.0, 1.0, time_fraction, reset_phase])
            else:
                observations.append(
                    [0.25 * self.distractors[env_idx][tick], 0.0, 0.0, time_fraction, reset_phase]
                )
        return observations


class OceanNoisyTMazeEnv:
    """Noisy cue + delayed branch decision task.

    This is deliberately harder than OceanTMazeEnv: the branch target is not
    shown exactly once. The agent receives several noisy evidence samples while
    it is still moving down the stem, then must choose left/right at the branch.
    """

    FORWARD = 0
    RIGHT = 1
    LEFT = 2

    def __init__(
        self,
        num_envs: int,
        size: int = 10,
        evidence_steps: int = 5,
        evidence_accuracy: float = 0.70,
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.size = max(6, size)
        self.evidence_steps = max(1, min(evidence_steps, self.size - 2))
        self.evidence_accuracy = min(1.0, max(0.5, evidence_accuracy))
        self.rng = random.Random(seed)
        self.states = [0 for _ in range(num_envs)]
        self.goals = [1 for _ in range(num_envs)]
        self.ticks = [0 for _ in range(num_envs)]
        self.evidence = [[0.0 for _ in range(self.evidence_steps)] for _ in range(num_envs)]
        self.distractors = [[0.0 for _ in range(self.size)] for _ in range(num_envs)]

    def reset(self) -> tuple[list[list[float]], list[dict[str, Any]]]:
        for env_idx in range(self.num_envs):
            self._reset_one(env_idx)
        return self._observations(), [{"env": "ocean_noisy_tmaze", "reset": True} for _ in range(self.num_envs)]

    def step(
        self, actions: Iterable[int]
    ) -> tuple[list[list[float]], list[float], list[bool], list[bool], list[dict[str, Any]]]:
        rewards: list[float] = []
        terminals: list[bool] = []
        truncations: list[bool] = []
        infos: list[dict[str, Any]] = []
        for env_idx, action in enumerate(actions):
            self.ticks[env_idx] += 1
            action = int(action) % 3
            reward = 0.0
            terminal = False
            truncation = False
            if self.states[env_idx] == self.size - 1:
                if action in (self.LEFT, self.RIGHT):
                    correct_action = self.RIGHT if self.goals[env_idx] > 0 else self.LEFT
                    reward = 1.0 if action == correct_action else 0.0
                    terminal = True
                    self._reset_one(env_idx)
                elif self.ticks[env_idx] > self.size + 4:
                    truncation = True
                    self._reset_one(env_idx)
            elif action == self.FORWARD:
                self.states[env_idx] += 1
            elif self.ticks[env_idx] > self.size + 4:
                truncation = True
                self._reset_one(env_idx)
            rewards.append(reward)
            terminals.append(terminal)
            truncations.append(truncation)
            infos.append({"env": "ocean_noisy_tmaze", "goal": self.goals[env_idx]})
        return self._observations(), rewards, terminals, truncations, infos

    def _reset_one(self, env_idx: int) -> None:
        goal = -1 if self.rng.randrange(2) == 0 else 1
        self.states[env_idx] = 0
        self.goals[env_idx] = goal
        self.ticks[env_idx] = 0
        self.evidence[env_idx] = [
            float(goal if self.rng.random() < self.evidence_accuracy else -goal)
            for _ in range(self.evidence_steps)
        ]
        self.distractors[env_idx] = [
            float(-1 if self.rng.randrange(2) == 0 else 1) for _ in range(self.size)
        ]

    def _observations(self) -> list[list[float]]:
        observations = []
        for env_idx, state in enumerate(self.states):
            reset_phase = 1.0 if state == 0 else 0.0
            time_fraction = state / max(1, self.size - 1)
            if state < self.evidence_steps:
                observations.append(
                    [self.evidence[env_idx][state], 1.0, 0.0, time_fraction, reset_phase]
                )
            elif state == self.size - 1:
                observations.append([0.0, 0.0, 1.0, time_fraction, reset_phase])
            else:
                observations.append(
                    [0.25 * self.distractors[env_idx][state], 0.0, 0.0, time_fraction, reset_phase]
                )
        return observations


class OceanSlotTMazeEnv:
    """Two-slot cue/query T-maze for explicit role-filler memory.

    Each episode writes two independent role values, waits through the corridor,
    then queries one role at the branch. A one-scalar memory can only keep the
    last cue; the intended SymLiquid state has to bind slot identity to value.
    """

    FORWARD = 0
    RIGHT = 1
    LEFT = 2

    def __init__(self, num_envs: int, size: int = 10, seed: int = 0):
        self.num_envs = num_envs
        self.size = max(6, size)
        self.rng = random.Random(seed)
        self.states = [0 for _ in range(num_envs)]
        self.slot_a = [1 for _ in range(num_envs)]
        self.slot_b = [-1 for _ in range(num_envs)]
        self.query_slot = [0 for _ in range(num_envs)]
        self.ticks = [0 for _ in range(num_envs)]
        self.distractors = [[0.0 for _ in range(self.size)] for _ in range(num_envs)]

    def reset(self) -> tuple[list[list[float]], list[dict[str, Any]]]:
        for env_idx in range(self.num_envs):
            self._reset_one(env_idx)
        return self._observations(), [{"env": "ocean_slot_tmaze", "reset": True} for _ in range(self.num_envs)]

    def step(
        self, actions: Iterable[int]
    ) -> tuple[list[list[float]], list[float], list[bool], list[bool], list[dict[str, Any]]]:
        rewards: list[float] = []
        terminals: list[bool] = []
        truncations: list[bool] = []
        infos: list[dict[str, Any]] = []
        for env_idx, action in enumerate(actions):
            self.ticks[env_idx] += 1
            action = int(action) % 3
            reward = 0.0
            terminal = False
            truncation = False
            if self.states[env_idx] == self.size - 1:
                if action in (self.LEFT, self.RIGHT):
                    target = self.slot_a[env_idx] if self.query_slot[env_idx] == 0 else self.slot_b[env_idx]
                    correct_action = self.RIGHT if target > 0 else self.LEFT
                    reward = 1.0 if action == correct_action else 0.0
                    terminal = True
                    self._reset_one(env_idx)
                elif self.ticks[env_idx] > self.size + 4:
                    truncation = True
                    self._reset_one(env_idx)
            elif action == self.FORWARD:
                self.states[env_idx] += 1
            elif self.ticks[env_idx] > self.size + 4:
                truncation = True
                self._reset_one(env_idx)
            rewards.append(reward)
            terminals.append(terminal)
            truncations.append(truncation)
            infos.append({"env": "ocean_slot_tmaze", "query_slot": self.query_slot[env_idx]})
        return self._observations(), rewards, terminals, truncations, infos

    def _reset_one(self, env_idx: int) -> None:
        self.states[env_idx] = 0
        self.ticks[env_idx] = 0
        self.slot_a[env_idx] = -1 if self.rng.randrange(2) == 0 else 1
        self.slot_b[env_idx] = -1 if self.rng.randrange(2) == 0 else 1
        self.query_slot[env_idx] = self.rng.randrange(2)
        self.distractors[env_idx] = [
            float(-1 if self.rng.randrange(2) == 0 else 1) for _ in range(self.size)
        ]

    def _observations(self) -> list[list[float]]:
        observations = []
        for env_idx, state in enumerate(self.states):
            reset_phase = 1.0 if state == 0 else 0.0
            time_fraction = state / max(1, self.size - 1)
            if state == 0:
                observations.append([float(self.slot_a[env_idx]), 1.0, 0.0, 0.0, 0.0, 0.0, time_fraction, reset_phase])
            elif state == 1:
                observations.append([float(self.slot_b[env_idx]), 0.0, 1.0, 0.0, 0.0, 0.0, time_fraction, reset_phase])
            elif state == self.size - 1:
                query_a = 1.0 if self.query_slot[env_idx] == 0 else 0.0
                query_b = 1.0 if self.query_slot[env_idx] == 1 else 0.0
                observations.append([0.0, 0.0, 0.0, 1.0, query_a, query_b, time_fraction, reset_phase])
            else:
                observations.append([0.25 * self.distractors[env_idx][state], 0.0, 0.0, 0.0, 0.0, 0.0, time_fraction, reset_phase])
        return observations


class OceanTMazeEnv:
    """Vectorized local port of PufferLib Ocean TMaze."""

    FORWARD = 0
    RIGHT = 1
    LEFT = 2

    def __init__(self, num_envs: int, size: int = 8, seed: int = 0):
        self.num_envs = num_envs
        self.size = max(3, size)
        self.rng = random.Random(seed)
        self.states = [0 for _ in range(num_envs)]
        self.starting_states = [2 for _ in range(num_envs)]
        self.ticks = [0 for _ in range(num_envs)]

    def reset(self) -> tuple[list[list[float]], list[dict[str, Any]]]:
        for env_idx in range(self.num_envs):
            self._reset_one(env_idx)
        return self._observations(), [{"env": "ocean_tmaze", "reset": True} for _ in range(self.num_envs)]

    def step(
        self, actions: Iterable[int]
    ) -> tuple[list[list[float]], list[float], list[bool], list[bool], list[dict[str, Any]]]:
        rewards: list[float] = []
        terminals: list[bool] = []
        truncations: list[bool] = []
        infos: list[dict[str, Any]] = []
        for env_idx, action in enumerate(actions):
            self.ticks[env_idx] += 1
            action = int(action) % 3
            reward = 0.0
            terminal = False
            truncation = False
            if self.states[env_idx] == self.size - 1:
                if action in (self.LEFT, self.RIGHT):
                    left_reward = 1.0 if self.starting_states[env_idx] == 2 else -1.0
                    right_reward = 1.0 if self.starting_states[env_idx] == 3 else -1.0
                    reward = left_reward if action == self.LEFT else right_reward
                    terminal = True
                    self._reset_one(env_idx)
                elif self.ticks[env_idx] > self.size + 4:
                    truncation = True
                    self._reset_one(env_idx)
            elif action == self.FORWARD:
                self.states[env_idx] += 1
            elif self.ticks[env_idx] > self.size + 4:
                truncation = True
                self._reset_one(env_idx)
            rewards.append(reward)
            terminals.append(terminal)
            truncations.append(truncation)
            infos.append({"env": "ocean_tmaze", "state": self.states[env_idx]})
        return self._observations(), rewards, terminals, truncations, infos

    def _reset_one(self, env_idx: int) -> None:
        self.states[env_idx] = 0
        self.starting_states[env_idx] = 2 + self.rng.randrange(2)
        self.ticks[env_idx] = 0

    def _observations(self) -> list[list[float]]:
        observations = []
        for state, starting_state in zip(self.states, self.starting_states):
            if state == 0:
                observations.append([float(starting_state), 1.0, 0.0, 0.0])
            elif state == self.size - 1:
                observations.append([1.0, 0.0, 1.0, 1.0])
            else:
                observations.append([1.0, 1.0, 0.0, 0.0])
        return observations


def numeric_hash_features(observation: Any, hv_dim: int) -> list[float]:
    sparse, norm = numeric_hash_sparse_features(observation, hv_dim)
    features = [0.0] * hv_dim
    for idx, value in sparse:
        features[idx] += value / norm
    return features


def artifact_sparse_features(
    observation: Any,
    artifact: SymLiquidPolicyArtifact,
    memory_state: list[float] | None = None,
    env_idx: int = 0,
) -> tuple[list[tuple[int, float]], float]:
    if artifact.feature_set == "cartpole_linear_v1":
        return cartpole_linear_sparse_features(observation, artifact.hv_dim)
    if artifact.feature_set == "dense_linear_v1":
        return dense_linear_sparse_features(observation, artifact.hv_dim)
    if artifact.feature_set == "memory_recurrent_linear_v1":
        return recurrent_linear_sparse_features(
            observation, artifact.hv_dim, memory_state, env_idx, mode="memory"
        )
    if artifact.feature_set == "evidence_recurrent_linear_v1":
        return recurrent_linear_sparse_features(
            observation, artifact.hv_dim, memory_state, env_idx, mode="evidence"
        )
    if artifact.feature_set == "evidence_sum_recurrent_linear_v1":
        return recurrent_linear_sparse_features(
            observation, artifact.hv_dim, memory_state, env_idx, mode="evidence_sum"
        )
    if artifact.feature_set == "evidence_tmaze_recurrent_linear_v1":
        return recurrent_linear_sparse_features(
            observation, artifact.hv_dim, memory_state, env_idx, mode="evidence_tmaze"
        )
    if artifact.feature_set == "evidence_sum_tmaze_recurrent_linear_v1":
        return recurrent_linear_sparse_features(
            observation, artifact.hv_dim, memory_state, env_idx, mode="evidence_sum_tmaze"
        )
    if artifact.feature_set == "slot_tmaze_recurrent_linear_v1":
        return slot_tmaze_sparse_features(observation, artifact.hv_dim, memory_state, env_idx)
    if artifact.feature_set == "tmaze_recurrent_linear_v1":
        return recurrent_linear_sparse_features(
            observation, artifact.hv_dim, memory_state, env_idx, mode="tmaze"
        )
    return numeric_hash_sparse_features(observation, artifact.hv_dim)


def numeric_hash_sparse_features(observation: Any, hv_dim: int) -> tuple[list[tuple[int, float]], float]:
    flat = flatten_numeric(observation)
    values: list[tuple[int, float]] = []
    add_sparse_feature(values, hv_dim, "bias", 1.0)
    for idx, value in enumerate(flat):
        bucketed = int(max(-16, min(16, round(value * 8.0))))
        add_sparse_feature(values, hv_dim, f"obs:{idx}:bucket:{bucketed}", 1.0)
        add_sparse_feature(values, hv_dim, f"obs:{idx}:sign:{1 if value >= 0 else -1}", 0.25)
    norm = math.sqrt(sum(value * value for _, value in values)) or 1.0
    return values, norm


def cartpole_linear_sparse_features(observation: Any, hv_dim: int) -> tuple[list[tuple[int, float]], float]:
    flat = flatten_numeric(observation)
    while len(flat) < 4:
        flat.append(0.0)
    x, x_dot, theta, theta_dot = flat[:4]
    dense = [
        1.0,
        x,
        x_dot,
        theta,
        theta_dot,
        x * x,
        theta * theta,
        x_dot * theta_dot,
        theta + 0.35 * theta_dot + 0.03 * x + 0.01 * x_dot,
    ]
    values = [(idx, float(value)) for idx, value in enumerate(dense[:hv_dim])]
    return values, 1.0


def dense_linear_sparse_features(observation: Any, hv_dim: int) -> tuple[list[tuple[int, float]], float]:
    flat = flatten_numeric(observation)
    dense = [1.0]
    dense.extend(flat)
    dense.extend(value * value for value in flat)
    if len(flat) >= 2:
        dense.append(flat[0] * flat[1])
    values = [(idx, float(value)) for idx, value in enumerate(dense[:hv_dim])]
    return values, 1.0


def recurrent_linear_sparse_features(
    observation: Any,
    hv_dim: int,
    memory_state: list[float] | None,
    env_idx: int,
    mode: str,
) -> tuple[list[tuple[int, float]], float]:
    flat = flatten_numeric(observation)
    if memory_state is None:
        memory_value = 0.0
    else:
        if len(memory_state) <= env_idx:
            memory_state.extend([0.0] * (env_idx + 1 - len(memory_state)))
        if flat:
            if mode == "memory" and abs(flat[0]) > 0.5:
                memory_state[env_idx] = 1.0 if flat[0] > 0.0 else -1.0
            elif mode in ("evidence", "evidence_tmaze", "evidence_sum", "evidence_sum_tmaze"):
                if len(flat) >= 5 and flat[4] > 0.5:
                    memory_state[env_idx] = 0.0
                if len(flat) >= 2 and flat[1] > 0.5:
                    if mode in ("evidence_sum", "evidence_sum_tmaze"):
                        memory_state[env_idx] += flat[0]
                    else:
                        memory_state[env_idx] = 0.75 * memory_state[env_idx] + flat[0]
            elif mode == "tmaze":
                if 1.75 <= flat[0] < 2.5:
                    memory_state[env_idx] = -1.0
                elif flat[0] >= 2.5:
                    memory_state[env_idx] = 1.0
        memory_value = memory_state[env_idx]

    dense = [1.0]
    dense.extend(flat)
    dense.append(memory_value)
    dense.extend(value * memory_value for value in flat)
    if mode == "tmaze":
        at_branch = 1.0 if len(flat) >= 4 and flat[1] == 0.0 and flat[2] > 0.5 and flat[3] > 0.5 else 0.0
        dense.append(at_branch)
        dense.append(at_branch * memory_value)
    elif mode in ("evidence", "evidence_tmaze", "evidence_sum", "evidence_sum_tmaze"):
        branch_or_decision_phase = flat[2] if len(flat) >= 3 else 0.0
        dense.append(memory_value * branch_or_decision_phase)
        dense.append((flat[0] if flat else 0.0) * memory_value)
        dense.append(branch_or_decision_phase)
    values = [(idx, float(value)) for idx, value in enumerate(dense[:hv_dim])]
    return values, 1.0


def slot_tmaze_sparse_features(
    observation: Any,
    hv_dim: int,
    memory_state: list[float] | None,
    env_idx: int,
) -> tuple[list[tuple[int, float]], float]:
    flat = flatten_numeric(observation)
    while len(flat) < 8:
        flat.append(0.0)
    cue, write_a, write_b, branch, query_a, query_b, time_fraction, reset_phase = flat[:8]
    if memory_state is None:
        slot_a = 0.0
        slot_b = 0.0
    else:
        offset = env_idx * 2
        if len(memory_state) <= offset + 1:
            memory_state.extend([0.0] * (offset + 2 - len(memory_state)))
        if reset_phase > 0.5:
            memory_state[offset] = 0.0
            memory_state[offset + 1] = 0.0
        if write_a > 0.5:
            memory_state[offset] = cue
        if write_b > 0.5:
            memory_state[offset + 1] = cue
        slot_a = memory_state[offset]
        slot_b = memory_state[offset + 1]
    selected = query_a * slot_a + query_b * slot_b
    dense = [
        1.0,
        cue,
        write_a,
        write_b,
        branch,
        query_a,
        query_b,
        time_fraction,
        reset_phase,
        slot_a,
        slot_b,
        selected,
        branch * selected,
        branch * slot_a,
        branch * slot_b,
        write_a * cue,
        write_b * cue,
        branch * query_a,
        branch * query_b,
        selected * time_fraction,
        slot_a * query_a,
        slot_b * query_b,
    ]
    return [(idx, float(value)) for idx, value in enumerate(dense[:hv_dim])], 1.0


def flatten_numeric(value: Any) -> list[float]:
    if hasattr(value, "flatten"):
        return [float(x) for x in value.flatten()]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numeric(item))
        return out
    return [0.0]


def add_feature(features: list[float], key: str, value: float) -> None:
    idx, signed = signed_feature(len(features), key, value)
    features[idx] += signed


def add_sparse_feature(features: list[tuple[int, float]], hv_dim: int, key: str, value: float) -> None:
    features.append(signed_feature(hv_dim, key, value))


def signed_feature(hv_dim: int, key: str, value: float) -> tuple[int, float]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:8], "little") % hv_dim
    sign = 1.0 if digest[8] & 1 == 0 else -1.0
    return idx, sign * value


def check() -> int:
    print(f"python {sys.version.split()[0]}")
    try:
        import pufferlib
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"pufferlib not installed: {exc}")
        return 0
    print(f"pufferlib import ok: {Path(pufferlib.__file__).resolve()}")
    print(f"pufferlib version: {getattr(pufferlib, '__version__', 'unknown')}")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch cuda device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # pragma: no cover - optional environment detail
        print(f"torch check failed: {exc}")
    try:
        import pufferlib._C as compiled_backend  # noqa: F401

        print("pufferlib compiled backend: ok")
    except Exception as exc:  # pragma: no cover - platform/build dependent
        print(f"pufferlib compiled backend unavailable: {exc}")
    return 0


def emit_json(payload: dict[str, Any], out_path: str | Path | None = None) -> None:
    text = json.dumps(payload, indent=2)
    print(text)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text + "\n", encoding="utf-8")


def write_json_file(payload: dict[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def numeric_summary(value: Any, *, first: int = 8) -> dict[str, Any]:
    flat = flatten_numeric(value)
    if not flat:
        return {"len": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "l1": 0.0, "first": []}
    return {
        "len": len(flat),
        "min": min(flat),
        "max": max(flat),
        "mean": sum(flat) / len(flat),
        "l1": sum(abs(x) for x in flat),
        "first": flat[:first],
    }


def policy_state_summary(policy: Any, env_idx: int) -> dict[str, Any]:
    if not isinstance(policy, SymLiquidPufferPolicy):
        return {"available": False, "reason": "policy_state_owned_by_rust_ffi_or_external_backend"}
    stride = memory_stride_for_feature_set(policy.artifact.feature_set)
    start = env_idx * stride
    values = policy._memory_state[start : start + stride]
    return {"available": True, "stride": stride, "summary": numeric_summary(values)}


def semantic_phase(env_name: str, observation: Any, info: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_numeric(observation)
    phase = "unknown"
    if env_name == "ocean-cartpole":
        theta = flat[2] if len(flat) > 2 else 0.0
        phase = "unsafe_angle" if abs(theta) > 0.12 else "balance"
    elif env_name in ("ocean-tmaze", "ocean-noisy-tmaze", "ocean-slot-tmaze"):
        at_branch = len(flat) > 2 and flat[2] > 0.5
        at_start = len(flat) > 1 and flat[1] <= 0.5
        if at_branch:
            phase = "branch_decision"
        elif at_start:
            phase = "cue_or_reset"
        else:
            phase = "corridor"
    elif env_name in ("ocean-memory", "ocean-noisy-memory"):
        phase = "query" if len(flat) > 1 and flat[1] > 0.5 else "cue_or_delay"
    elif env_name == "ocean-chain":
        phase = "chain_progress"
    return {
        "phase": phase,
        "info_state": info.get("state") if isinstance(info, dict) else None,
        "observation_summary": numeric_summary(flat),
    }


def init_eventized_rollout_log(
    *,
    env_name: str,
    artifact_path: str | Path,
    artifact: SymLiquidPolicyArtifact,
    num_envs: int,
    steps: int,
    policy_backend: str,
    env_limit: int,
    step_limit: int,
    governance_reflex: bool,
) -> dict[str, Any]:
    return {
        "policy": "local_only_no_external_inference",
        "methodology": "high_bandwidth_embodied_rollout_logging",
        "schema_version": "0.1.0",
        "env": env_name,
        "artifact": str(artifact_path),
        "feature_set": artifact.feature_set,
        "num_envs": num_envs,
        "steps": steps,
        "policy_backend": policy_backend,
        "governance_reflex": governance_reflex,
        "retention_policy": {
            "raw_stream": "bounded_sampled_envs_and_steps",
            "sampled_envs": min(num_envs, env_limit),
            "sampled_steps": min(steps, step_limit),
            "always_count_events": True,
            "retain_anomalies": [
                "negative_reward",
                "terminal_without_positive_reward",
                "truncation",
                "reflex_override",
            ],
        },
        "streams": {
            "raw_windows": [],
            "event_log": [],
            "semantic_trace": [],
            "skill_trace": [],
            "residual_log": [],
        },
        "summary": {
            "sampled_raw_windows": 0,
            "event_count": 0,
            "semantic_events": 0,
            "skill_events": 0,
            "residual_events": 0,
            "external_inference_calls": 0,
        },
    }


def append_rollout_event(
    log: dict[str, Any],
    *,
    step: int,
    env_idx: int,
    env_name: str,
    observation: Any,
    next_observation: Any,
    proposed_action: int,
    action: int,
    reward: float,
    terminal: bool,
    truncation: bool,
    info: dict[str, Any],
    policy: Any,
    reflex_override: bool,
) -> None:
    retention = log["retention_policy"]
    sample_raw = step < retention["sampled_steps"] and env_idx < retention["sampled_envs"]
    mode = "reflex_failsafe" if reflex_override else "compiled_tool"
    if sample_raw:
        log["streams"]["raw_windows"].append(
            {
                "step": step,
                "env_idx": env_idx,
                "observation": observation,
                "next_observation": next_observation,
                "proposed_action": proposed_action,
                "action": action,
                "reward": reward,
                "terminal": terminal,
                "truncation": truncation,
                "info": info,
            }
        )
        log["summary"]["sampled_raw_windows"] += 1

        log["streams"]["semantic_trace"].append(
            {
                "step": step,
                "env_idx": env_idx,
                **semantic_phase(env_name, observation, info),
            }
        )
        log["summary"]["semantic_events"] += 1

        log["streams"]["skill_trace"].append(
            {
                "step": step,
                "env_idx": env_idx,
                "execution_mode": mode,
                "policy_backend": log["policy_backend"],
                "feature_set": log["feature_set"],
                "proposed_action": proposed_action,
                "action": action,
                "state_summary": policy_state_summary(policy, env_idx),
            }
        )
        log["summary"]["skill_events"] += 1

    event_kinds = []
    if reward > 0:
        event_kinds.append("positive_reward")
    if reward < 0:
        event_kinds.append("negative_reward")
    if terminal:
        event_kinds.append("terminal")
    if truncation:
        event_kinds.append("truncation")
    if reflex_override:
        event_kinds.append("reflex_override")
    for kind in event_kinds:
        log["streams"]["event_log"].append(
            {
                "step": step,
                "env_idx": env_idx,
                "kind": kind,
                "reward": reward,
                "action": action,
                "execution_mode": mode,
            }
        )
        log["summary"]["event_count"] += 1

    residual_kinds = []
    if reward < 0:
        residual_kinds.append("negative_reward")
    if terminal and reward <= 0:
        residual_kinds.append("terminal_without_positive_reward")
    if truncation:
        residual_kinds.append("truncation")
    if reflex_override:
        residual_kinds.append("reflex_override")
    for kind in residual_kinds:
        log["streams"]["residual_log"].append(
            {
                "step": step,
                "env_idx": env_idx,
                "kind": kind,
                "reward": reward,
                "proposed_action": proposed_action,
                "action": action,
                "semantic_phase": semantic_phase(env_name, observation, info)["phase"],
            }
        )
        log["summary"]["residual_events"] += 1


def rollout_smoke(
    artifact_path: str | Path,
    env_name: str,
    num_envs: int,
    steps: int,
    obs_dim: int,
    action_modulo: int,
    governance_reflex: bool = False,
    use_rust_ffi: bool = False,
    out_path: str | Path | None = None,
    event_log_out: str | Path | None = None,
    event_log_env_limit: int = 4,
    event_log_step_limit: int = 64,
) -> int:
    artifact = SymLiquidPolicyArtifact.load(artifact_path)
    policy_backend = "rust_ffi" if use_rust_ffi else "python"
    policy = RustFfiPolicy(artifact_path, num_envs) if use_rust_ffi else SymLiquidPufferPolicy(artifact)
    env, action_modulo = make_local_env(
        env_name, num_envs=num_envs, obs_dim=obs_dim, action_modulo=action_modulo, seed=0
    )
    observations, _infos = env.reset()
    total_reward = 0.0
    total_dones = 0
    total_truncations = 0
    reflex_overrides = 0
    positive_rewards = 0
    negative_rewards = 0
    event_log = None
    if event_log_out:
        event_log = init_eventized_rollout_log(
            env_name=env_name,
            artifact_path=artifact_path,
            artifact=artifact,
            num_envs=num_envs,
            steps=steps,
            policy_backend=policy_backend,
            env_limit=event_log_env_limit,
            step_limit=event_log_step_limit,
            governance_reflex=governance_reflex,
        )
    start = time.perf_counter()
    for step_idx in range(steps):
        observations_before = observations
        proposed_actions = policy.act(observations_before)
        actions = [action % action_modulo for action in proposed_actions]
        reflex_flags = [False] * len(actions)
        if governance_reflex and env_name == "ocean-cartpole":
            governed_actions = [
                cartpole_reflex_action(obs, action)
                for obs, action in zip(observations_before, actions)
            ]
            reflex_flags = [
                int(before) != int(after) for before, after in zip(actions, governed_actions)
            ]
            reflex_overrides += sum(1 for flag in reflex_flags if flag)
            actions = governed_actions
        observations, rewards, terminals, _truncations, _infos = env.step(actions)
        total_reward += sum(rewards)
        positive_rewards += sum(1 for reward in rewards if reward > 0)
        negative_rewards += sum(1 for reward in rewards if reward < 0)
        total_dones += sum(1 for done in terminals if done)
        total_truncations += sum(1 for done in _truncations if done)
        if event_log is not None:
            for env_idx, (
                before,
                after,
                proposed_action,
                action,
                reward,
                terminal,
                truncation,
                info,
                reflex_flag,
            ) in enumerate(
                zip(
                    observations_before,
                    observations,
                    proposed_actions,
                    actions,
                    rewards,
                    terminals,
                    _truncations,
                    _infos,
                    reflex_flags,
                )
            ):
                append_rollout_event(
                    event_log,
                    step=step_idx,
                    env_idx=env_idx,
                    env_name=env_name,
                    observation=before,
                    next_observation=after,
                    proposed_action=int(proposed_action % action_modulo),
                    action=int(action),
                    reward=float(reward),
                    terminal=bool(terminal),
                    truncation=bool(truncation),
                    info=info if isinstance(info, dict) else {},
                    policy=policy,
                    reflex_override=bool(reflex_flag),
                )
    elapsed = max(time.perf_counter() - start, 1.0e-9)
    transitions = num_envs * steps
    if event_log is not None and event_log_out is not None:
        event_log["summary"].update(
            {
                "transitions": transitions,
                "total_reward": total_reward,
                "mean_reward": total_reward / max(1, transitions),
                "normalized_perf": normalized_perf(
                    env_name, total_reward, transitions, total_dones
                ),
                "positive_rewards": positive_rewards,
                "negative_rewards": negative_rewards,
                "dones": total_dones,
                "truncations": total_truncations,
                "reflex_overrides": reflex_overrides,
            }
        )
        write_json_file(event_log, event_log_out)
    emit_json(
        {
            "env": env_name,
            "num_envs": num_envs,
            "steps": steps,
            "transitions": transitions,
            "transitions_per_second": transitions / elapsed,
            "total_reward": total_reward,
            "mean_reward": total_reward / max(1, transitions),
            "normalized_perf": normalized_perf(env_name, total_reward, transitions, total_dones),
            "positive_rewards": positive_rewards,
            "negative_rewards": negative_rewards,
            "dones": total_dones,
            "truncations": total_truncations,
            "artifact": str(artifact_path),
            "feature_set": artifact.feature_set,
            "policy_backend": policy_backend,
            "ffi_library": policy.library_path if isinstance(policy, RustFfiPolicy) else None,
            "governance_reflex": governance_reflex,
            "reflex_overrides": reflex_overrides,
            "event_log": str(event_log_out) if event_log_out else None,
            "external_inference_calls": 0,
        },
        out_path,
    )
    if isinstance(policy, RustFfiPolicy):
        policy.close()
    return 0


def normalized_perf(env_name: str, total_reward: float, transitions: int, dones: int) -> float:
    if env_name == "ocean-chain":
        return min(1.0, max(0.0, total_reward / max(1.0, transitions * 0.46875)))
    if env_name == "ocean-memory":
        return min(1.0, max(0.0, total_reward / max(1.0, transitions / 8.0)))
    if env_name == "ocean-noisy-memory":
        return min(1.0, max(0.0, total_reward / max(1.0, transitions / 12.0)))
    if env_name == "ocean-noisy-tmaze":
        return min(1.0, max(0.0, total_reward / max(1.0, transitions / 10.0)))
    if env_name == "ocean-slot-tmaze":
        return min(1.0, max(0.0, total_reward / max(1.0, transitions / 10.0)))
    if env_name == "ocean-tmaze":
        return min(1.0, max(0.0, (total_reward + max(1, dones)) / max(1.0, 2.0 * max(1, dones))))
    return total_reward / max(1, transitions)


def make_local_env(
    env_name: str,
    num_envs: int,
    obs_dim: int = 8,
    action_modulo: int = 4,
    seed: int = 0,
) -> tuple[Any, int]:
    if env_name == "ocean-cartpole":
        return OceanCartpoleEnv(num_envs=num_envs, seed=seed), 2
    if env_name == "ocean-chain":
        return OceanChainMdpEnv(num_envs=num_envs, seed=seed), 2
    if env_name == "ocean-memory":
        return OceanMemoryEnv(num_envs=num_envs, seed=seed), 2
    if env_name == "ocean-noisy-memory":
        return OceanNoisyMemoryEnv(num_envs=num_envs, seed=seed), 2
    if env_name == "ocean-noisy-tmaze":
        return OceanNoisyTMazeEnv(num_envs=num_envs, seed=seed), 3
    if env_name == "ocean-slot-tmaze":
        return OceanSlotTMazeEnv(num_envs=num_envs, seed=seed), 3
    if env_name == "ocean-tmaze":
        return OceanTMazeEnv(num_envs=num_envs, seed=seed), 3
    if env_name == "toy":
        return PufferStyleToyEnv(num_envs=num_envs, obs_dim=obs_dim, action_modulo=action_modulo), action_modulo
    raise ValueError(
        f"unknown rollout env {env_name!r}; expected toy, ocean-cartpole, ocean-chain, ocean-memory, ocean-noisy-memory, ocean-noisy-tmaze, ocean-slot-tmaze, or ocean-tmaze"
    )


def cartpole_reflex_action(observation: Any, proposed_action: int) -> int:
    """Low-latency local safety reflex for Ocean CartPole-style smoke tests.

    This is not an external model call and not a learned-policy score. It is a
    transparent reflex/failsafe overlay used to test the CLC execution-mode
    boundary: policy proposes, reflex may override when the pole state is
    already outside a small safety margin.
    """

    flat = flatten_numeric(observation)
    if len(flat) < 4:
        return int(proposed_action) % 2
    x, x_dot, theta, theta_dot = flat[:4]
    urgency = abs(theta) + 0.35 * abs(theta_dot) + 0.05 * abs(x) + 0.02 * abs(x_dot)
    if urgency < 0.025:
        return int(proposed_action) % 2
    return 1 if theta + 0.35 * theta_dot + 0.03 * x + 0.01 * x_dot > 0.0 else 0


def train_cartpole_policy(
    policy_out: str | Path,
    report_out: str | Path | None,
    iterations: int,
    population: int,
    elite_count: int,
    num_envs: int,
    train_steps: int,
    eval_steps: int,
    seed: int,
) -> int:
    """Train a small local CartPole policy with cross-entropy search.

    This is deliberately simple and fully local: no torch model, no hosted
    inference, and no reward-model calls. It produces a SymLiquid-compatible
    artifact so the same Puffer/Ocean adapter path can evaluate it.
    """

    hv_dim = 9
    rng = random.Random(seed)
    elite_count = max(1, min(elite_count, population))
    param_dim = hv_dim + 1
    mean = [0.0 for _ in range(param_dim)]
    std = [1.0 for _ in range(param_dim)]
    best_params = list(mean)
    best_score = float("-inf")
    history: list[dict[str, Any]] = []

    train_seeds = [seed + 101 * idx for idx in range(3)]
    for iteration in range(iterations):
        candidates = []
        for _ in range(population):
            params = [
                rng.gauss(mu, sigma)
                for mu, sigma in zip(mean, std)
            ]
            score = evaluate_cartpole_params(
                params,
                num_envs=num_envs,
                steps=train_steps,
                seeds=train_seeds,
            )
            candidates.append((score, params))
        candidates.sort(key=lambda item: item[0], reverse=True)
        elites = candidates[:elite_count]
        if elites[0][0] > best_score:
            best_score = elites[0][0]
            best_params = list(elites[0][1])
        for idx in range(param_dim):
            values = [params[idx] for _, params in elites]
            mu = sum(values) / len(values)
            variance = sum((value - mu) ** 2 for value in values) / len(values)
            mean[idx] = mu
            std[idx] = max(0.03, math.sqrt(variance) * 0.9)
        history.append(
            {
                "iteration": iteration,
                "best_mean_reward": elites[0][0],
                "elite_mean_reward": sum(score for score, _ in elites) / len(elites),
                "search_std_mean": sum(std) / len(std),
            }
        )

    eval_seeds = [seed + 10_000 + 211 * idx for idx in range(5)]
    eval_score = evaluate_cartpole_params(
        best_params,
        num_envs=num_envs,
        steps=eval_steps,
        seeds=eval_seeds,
    )
    artifact = cartpole_params_to_artifact(best_params, hv_dim)
    policy_path = Path(policy_out)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    report = {
        "algorithm": "local_cross_entropy_search",
        "env": "ocean-cartpole",
        "policy_out": str(policy_path),
        "iterations": iterations,
        "population": population,
        "elite_count": elite_count,
        "num_envs": num_envs,
        "train_steps": train_steps,
        "eval_steps": eval_steps,
        "seed": seed,
        "best_train_mean_reward": best_score,
        "eval_mean_reward": eval_score,
        "feature_set": artifact["feature_set"],
        "external_inference_calls": 0,
        "history": history,
    }
    emit_json(report, report_out)
    return 0


def evaluate_cartpole_params(
    params: list[float], num_envs: int, steps: int, seeds: list[int]
) -> float:
    scores = []
    for seed in seeds:
        env = OceanCartpoleEnv(num_envs=num_envs, seed=seed)
        observations, _infos = env.reset()
        total_reward = 0.0
        transitions = max(1, num_envs * steps)
        for _ in range(steps):
            actions = [cartpole_param_action(params, obs) for obs in observations]
            observations, rewards, _terminals, _truncations, _infos = env.step(actions)
            total_reward += sum(rewards)
        scores.append(total_reward / transitions)
    return sum(scores) / len(scores)


def cartpole_param_action(params: list[float], observation: Any) -> int:
    sparse, _norm = cartpole_linear_sparse_features(observation, len(params) - 1)
    logit = params[-1]
    for idx, value in sparse:
        logit += params[idx] * value
    return 1 if logit >= 0.0 else 0


def cartpole_params_to_artifact(params: list[float], hv_dim: int) -> dict[str, Any]:
    weights = [0.0 for _ in range(2 * hv_dim)]
    bias = [0.0, params[-1]]
    for idx, value in enumerate(params[:hv_dim]):
        weights[hv_dim + idx] = value
    return {
        "labels": ["action_0", "action_1"],
        "hv_dim": hv_dim,
        "output_dim": 2,
        "weights": weights,
        "bias": bias,
        "feature_set": "cartpole_linear_v1",
        "training": {
            "algorithm": "local_cross_entropy_search",
            "external_inference_calls": 0,
        },
    }


def train_discrete_policy(
    env_name: str,
    policy_out: str | Path,
    report_out: str | Path | None,
    iterations: int,
    population: int,
    elite_count: int,
    num_envs: int,
    train_steps: int,
    eval_steps: int,
    seed: int,
    use_rust_ffi: bool = False,
) -> int:
    action_modulo = local_env_action_modulo(env_name)
    feature_set, hv_dim = policy_shape_for_env(env_name)
    rng = random.Random(seed)
    elite_count = max(1, min(elite_count, population))
    param_dim = action_modulo * hv_dim + action_modulo
    mean = initial_discrete_policy_params(env_name, hv_dim, action_modulo)
    std = [0.5 for _ in range(param_dim)]
    best_params = list(mean)
    best_score = float("-inf")
    history: list[dict[str, Any]] = []
    train_seeds = [seed + 101 * idx for idx in range(3)]

    for iteration in range(iterations):
        candidates = []
        for _ in range(population):
            params = [rng.gauss(mu, sigma) for mu, sigma in zip(mean, std)]
            score = evaluate_discrete_params(
                params,
                env_name=env_name,
                feature_set=feature_set,
                hv_dim=hv_dim,
                output_dim=action_modulo,
                num_envs=num_envs,
                steps=train_steps,
                seeds=train_seeds,
                use_rust_ffi=use_rust_ffi,
            )
            candidates.append((score, params))
        candidates.sort(key=lambda item: item[0], reverse=True)
        elites = candidates[:elite_count]
        if elites[0][0] > best_score:
            best_score = elites[0][0]
            best_params = list(elites[0][1])
        for idx in range(param_dim):
            values = [params[idx] for _, params in elites]
            mu = sum(values) / len(values)
            variance = sum((value - mu) ** 2 for value in values) / len(values)
            mean[idx] = mu
            std[idx] = max(0.03, math.sqrt(variance) * 0.9)
        history.append(
            {
                "iteration": iteration,
                "best_mean_reward": elites[0][0],
                "elite_mean_reward": sum(score for score, _ in elites) / len(elites),
                "search_std_mean": sum(std) / len(std),
            }
        )

    eval_seeds = [seed + 10_000 + 211 * idx for idx in range(5)]
    eval_score = evaluate_discrete_params(
        best_params,
        env_name=env_name,
        feature_set=feature_set,
        hv_dim=hv_dim,
        output_dim=action_modulo,
        num_envs=num_envs,
        steps=eval_steps,
        seeds=eval_seeds,
        use_rust_ffi=use_rust_ffi,
    )
    artifact = discrete_params_to_artifact(best_params, hv_dim, action_modulo, feature_set, env_name)
    policy_path = Path(policy_out)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    report = {
        "algorithm": "local_cross_entropy_search",
        "env": env_name,
        "policy_out": str(policy_path),
        "iterations": iterations,
        "population": population,
        "elite_count": elite_count,
        "num_envs": num_envs,
        "train_steps": train_steps,
        "eval_steps": eval_steps,
        "seed": seed,
        "best_train_mean_reward": best_score,
        "eval_mean_reward": eval_score,
        "feature_set": feature_set,
        "policy_backend": "rust_ffi" if use_rust_ffi else "python",
        "initialization": "cgs_governance_prior",
        "external_inference_calls": 0,
        "history": history,
    }
    emit_json(report, report_out)
    return 0


def train_discrete_policy_rust_ffi(
    env_name: str,
    policy_out: str | Path,
    report_out: str | Path | None,
    iterations: int,
    population: int,
    elite_count: int,
    num_envs: int,
    train_steps: int,
    eval_steps: int,
    seed: int,
) -> int:
    _lib_path, lib = load_rust_ffi_library()
    lib.symliquid_train_discrete_cem.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint64,
    ]
    lib.symliquid_train_discrete_cem.restype = ctypes.c_int
    policy_path = Path(policy_out)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    report_arg = None
    if report_out:
        report_path = Path(report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_arg = str(report_path).encode("utf-8")
    status = lib.symliquid_train_discrete_cem(
        env_name.encode("utf-8"),
        str(policy_path).encode("utf-8"),
        report_arg,
        ctypes.c_size_t(iterations),
        ctypes.c_size_t(population),
        ctypes.c_size_t(elite_count),
        ctypes.c_size_t(num_envs),
        ctypes.c_size_t(train_steps),
        ctypes.c_size_t(eval_steps),
        ctypes.c_uint64(seed),
    )
    if status != 0:
        raise RuntimeError(f"Rust FFI rollout trainer failed with status {status}")
    if report_out:
        print(Path(report_out).read_text(encoding="utf-8").rstrip())
    return 0


def evaluate_discrete_params(
    params: list[float],
    env_name: str,
    feature_set: str,
    hv_dim: int,
    output_dim: int,
    num_envs: int,
    steps: int,
    seeds: list[int],
    use_rust_ffi: bool = False,
) -> float:
    scores = []
    artifact = SymLiquidPolicyArtifact(
        labels=[f"action_{idx}" for idx in range(output_dim)],
        hv_dim=hv_dim,
        output_dim=output_dim,
        weights=params[: output_dim * hv_dim],
        bias=params[output_dim * hv_dim :],
        feature_set=feature_set,
    )
    ffi_policy: RustFfiPolicy | None = None
    try:
        if use_rust_ffi:
            ffi_policy = RustFfiPolicy.from_params(
                feature_set=feature_set,
                hv_dim=hv_dim,
                output_dim=output_dim,
                weights=artifact.weights,
                bias=artifact.bias,
                num_envs=num_envs,
            )
        for seed in seeds:
            env, action_modulo = make_local_env(env_name, num_envs=num_envs, seed=seed)
            if ffi_policy is None:
                policy = SymLiquidPufferPolicy(artifact)
            else:
                ffi_policy.reset(num_envs)
                policy = ffi_policy
            observations, _infos = env.reset()
            total_reward = 0.0
            transitions = max(1, num_envs * steps)
            for _ in range(steps):
                actions = [action % action_modulo for action in policy.act(observations)]
                observations, rewards, _terminals, _truncations, _infos = env.step(actions)
                total_reward += sum(rewards)
            scores.append(total_reward / transitions)
    finally:
        if ffi_policy is not None:
            ffi_policy.close()
    return sum(scores) / len(scores)


def local_env_action_modulo(env_name: str) -> int:
    return 3 if env_name in ("ocean-tmaze", "ocean-noisy-tmaze", "ocean-slot-tmaze") else 2


def policy_shape_for_env(env_name: str) -> tuple[str, int]:
    if env_name == "ocean-memory":
        return "memory_recurrent_linear_v1", 5
    if env_name == "ocean-noisy-memory":
        return "evidence_sum_recurrent_linear_v1", 15
    if env_name == "ocean-noisy-tmaze":
        return "evidence_sum_tmaze_recurrent_linear_v1", 15
    if env_name == "ocean-slot-tmaze":
        return "slot_tmaze_recurrent_linear_v1", 22
    if env_name == "ocean-tmaze":
        return "tmaze_recurrent_linear_v1", 12
    return "dense_linear_v1", 8


def discrete_params_to_artifact(
    params: list[float],
    hv_dim: int,
    output_dim: int,
    feature_set: str,
    env_name: str,
) -> dict[str, Any]:
    return {
        "labels": [f"action_{idx}" for idx in range(output_dim)],
        "hv_dim": hv_dim,
        "output_dim": output_dim,
        "weights": params[: output_dim * hv_dim],
        "bias": params[output_dim * hv_dim :],
        "feature_set": feature_set,
        "training": {
            "algorithm": "local_cross_entropy_search",
            "env": env_name,
            "initialization": "cgs_governance_prior",
            "external_inference_calls": 0,
        },
    }


def initial_discrete_policy_params(env_name: str, hv_dim: int, output_dim: int) -> list[float]:
    params = [0.0 for _ in range(output_dim * hv_dim + output_dim)]
    bias_offset = output_dim * hv_dim
    if env_name == "ocean-chain":
        params[bias_offset + 1] = 2.0
    elif env_name == "ocean-memory":
        memory_idx = 2
        params[0 * hv_dim + memory_idx] = -2.0
        params[1 * hv_dim + memory_idx] = 2.0
    elif env_name == "ocean-noisy-memory":
        decision_memory_idx = 12
        params[0 * hv_dim + decision_memory_idx] = -2.0
        params[1 * hv_dim + decision_memory_idx] = 2.0
    elif env_name == "ocean-noisy-tmaze":
        branch_memory_idx = 12
        branch_phase_idx = 14
        params[bias_offset + OceanNoisyTMazeEnv.FORWARD] = 1.0
        params[OceanNoisyTMazeEnv.FORWARD * hv_dim + branch_phase_idx] = -2.0
        params[bias_offset + OceanNoisyTMazeEnv.RIGHT] = -1.0
        params[OceanNoisyTMazeEnv.RIGHT * hv_dim + branch_phase_idx] = 1.0
        params[OceanNoisyTMazeEnv.RIGHT * hv_dim + branch_memory_idx] = 2.0
        params[bias_offset + OceanNoisyTMazeEnv.LEFT] = -1.0
        params[OceanNoisyTMazeEnv.LEFT * hv_dim + branch_phase_idx] = 1.0
        params[OceanNoisyTMazeEnv.LEFT * hv_dim + branch_memory_idx] = -2.0
    elif env_name == "ocean-slot-tmaze":
        branch_phase_idx = 4
        selected_memory_idx = 12
        params[bias_offset + OceanSlotTMazeEnv.FORWARD] = 1.0
        params[OceanSlotTMazeEnv.FORWARD * hv_dim + branch_phase_idx] = -2.0
        params[bias_offset + OceanSlotTMazeEnv.RIGHT] = -1.0
        params[OceanSlotTMazeEnv.RIGHT * hv_dim + branch_phase_idx] = 1.0
        params[OceanSlotTMazeEnv.RIGHT * hv_dim + selected_memory_idx] = 2.0
        params[bias_offset + OceanSlotTMazeEnv.LEFT] = -1.0
        params[OceanSlotTMazeEnv.LEFT * hv_dim + branch_phase_idx] = 1.0
        params[OceanSlotTMazeEnv.LEFT * hv_dim + selected_memory_idx] = -2.0
    elif env_name == "ocean-tmaze":
        at_branch_idx = 10
        branch_memory_idx = 11
        params[bias_offset + OceanTMazeEnv.FORWARD] = 1.0
        params[OceanTMazeEnv.FORWARD * hv_dim + at_branch_idx] = -2.0
        params[bias_offset + OceanTMazeEnv.RIGHT] = -1.0
        params[OceanTMazeEnv.RIGHT * hv_dim + at_branch_idx] = 1.0
        params[OceanTMazeEnv.RIGHT * hv_dim + branch_memory_idx] = 2.0
        params[bias_offset + OceanTMazeEnv.LEFT] = -1.0
        params[OceanTMazeEnv.LEFT * hv_dim + at_branch_idx] = 1.0
        params[OceanTMazeEnv.LEFT * hv_dim + branch_memory_idx] = -2.0
    return params


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--artifact")
    parser.add_argument("--train-cartpole-policy", action="store_true")
    parser.add_argument("--train-discrete-policy", action="store_true")
    parser.add_argument("--policy-out", default="reports/symliquid_ocean_cartpole_policy.json")
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--train-steps", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-actions", type=int, default=0)
    parser.add_argument("--rollout-smoke-steps", type=int, default=0)
    parser.add_argument(
        "--env",
        choices=[
            "toy",
            "ocean-cartpole",
            "ocean-chain",
            "ocean-memory",
            "ocean-noisy-memory",
            "ocean-noisy-tmaze",
            "ocean-slot-tmaze",
            "ocean-tmaze",
        ],
        default="toy",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--action-modulo", type=int, default=4)
    parser.add_argument("--governance-reflex", action="store_true")
    parser.add_argument("--use-rust-ffi", action="store_true")
    parser.add_argument("--event-log-out")
    parser.add_argument("--event-log-env-limit", type=int, default=4)
    parser.add_argument("--event-log-step-limit", type=int, default=64)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.check:
        return check()
    if args.train_cartpole_policy:
        return train_cartpole_policy(
            policy_out=args.policy_out,
            report_out=args.out,
            iterations=args.iterations,
            population=args.population,
            elite_count=args.elite_count,
            num_envs=args.num_envs,
            train_steps=args.train_steps,
            eval_steps=args.eval_steps,
            seed=args.seed,
        )
    if args.train_discrete_policy:
        if args.use_rust_ffi:
            return train_discrete_policy_rust_ffi(
                env_name=args.env,
                policy_out=args.policy_out,
                report_out=args.out,
                iterations=args.iterations,
                population=args.population,
                elite_count=args.elite_count,
                num_envs=args.num_envs,
                train_steps=args.train_steps,
                eval_steps=args.eval_steps,
                seed=args.seed,
            )
        return train_discrete_policy(
            env_name=args.env,
            policy_out=args.policy_out,
            report_out=args.out,
            iterations=args.iterations,
            population=args.population,
            elite_count=args.elite_count,
            num_envs=args.num_envs,
            train_steps=args.train_steps,
            eval_steps=args.eval_steps,
            seed=args.seed,
            use_rust_ffi=args.use_rust_ffi,
        )
    if args.artifact and args.smoke_actions:
        artifact = SymLiquidPolicyArtifact.load(args.artifact)
        policy = (
            RustFfiPolicy(args.artifact, args.smoke_actions)
            if args.use_rust_ffi
            else SymLiquidPufferPolicy(artifact)
        )
        observations = [
            [float(idx % 5) / 4.0, float((idx * 3) % 7) / 6.0]
            for idx in range(args.smoke_actions)
        ]
        actions = policy.act(observations)
        emit_json(
            {
                "actions": actions,
                "count": len(actions),
                "policy_backend": "rust_ffi" if args.use_rust_ffi else "python",
            },
            args.out,
        )
        if isinstance(policy, RustFfiPolicy):
            policy.close()
        return 0
    if args.artifact and args.rollout_smoke_steps:
        return rollout_smoke(
            args.artifact,
            env_name=args.env,
            num_envs=args.num_envs,
            steps=args.rollout_smoke_steps,
            obs_dim=args.obs_dim,
            action_modulo=args.action_modulo,
            governance_reflex=args.governance_reflex,
            use_rust_ffi=args.use_rust_ffi,
            out_path=args.out,
            event_log_out=args.event_log_out,
            event_log_env_limit=args.event_log_env_limit,
            event_log_step_limit=args.event_log_step_limit,
        )
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
