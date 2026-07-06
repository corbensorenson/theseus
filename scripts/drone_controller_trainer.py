"""Train and evaluate tiny local drone controllers for pressure-runner use.

This is intentionally small and dependency-light. It is not a final racing
policy trainer; it gives Project Theseus a real local learning loop behind
drone RL pressure cards so the autonomy loop can improve, score, escrow
residuals, and escalate architecture walls from evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["gym_pybullet_drones", "pyflyt", "pyflyt_waypoints"],
        default="gym_pybullet_drones",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--target-z", type=float, default=1.0)
    parser.add_argument("--policy-out", default="")
    parser.add_argument("--trace-out", default="")
    parser.add_argument("--replay-path", default="")
    parser.add_argument("--transfer-artifact-out", default="")
    parser.add_argument("--eval-seed-count", type=int, default=3)
    parser.add_argument("--min-train-candidate-evals", type=int, default=0)
    parser.add_argument("--min-train-env-steps", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    learner_memory = load_persistent_memory(args)
    env_contract = probe_env_contract(args.source, args.seed)
    best, history, replay_rows = train_controller(args, rng, learner_memory)
    replay_path = resolve_replay_path(args)
    append_replay_rows(replay_path, replay_rows)
    replay_summary = replay_buffer_summary(replay_path, args.source)
    budget = budget_accounting(args, replay_rows)
    trace_path = resolve_trace_path(args) if args.trace_out or args.out else None
    eval_seeds = deterministic_eval_seeds(args)
    evaluation = evaluate_controller(
        best["params"],
        args,
        episodes=max(args.episodes, len(eval_seeds)),
        seed_offset=10_000,
        seed_sequence=eval_seeds,
        trace_path=trace_path,
    )
    residuals = residuals_for_evaluation(evaluation)
    policy_path = Path(args.policy_out) if args.policy_out else (
        ROOT / "reports" / f"drone_controller_{args.source}_seed{args.seed}.json"
    )
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = {
        "policy": "theseus_tiny_drone_controller_v0",
        "created_utc": now(),
        "source": args.source,
        "seed": args.seed,
        "target_z": args.target_z,
        "controller": controller_name(args.source),
        "learner": "persistent_replay_elite_policy_search_v1",
        "params": [round(float(value), 8) for value in best["params"]],
        "state_normalizer": state_normalizer_spec(args.source),
        "action_adapter": action_adapter_spec(env_contract),
        "deterministic_eval_seeds": eval_seeds,
        "training_best_score": best["score"],
        "evaluation_score": evaluation["score"],
        "env_contract": env_contract,
        "replay_path": rel_or_abs(replay_path),
        "replay_summary": replay_summary,
        "budget": budget,
        "external_inference_calls": 0,
    }
    write_json(policy_path, policy)
    transfer_artifact_path = write_drone_transfer_artifact(args, policy_path, policy, evaluation, residuals)
    report = {
        "policy": "local_only_no_external_inference",
        "methodology": "theseus_tiny_drone_controller_trainer_v0",
        "created_utc": now(),
        "source": args.source,
        "seed": args.seed,
        "episodes": args.episodes,
        "steps": args.steps,
        "iterations": args.iterations,
        "population": args.population,
        "elite_count": args.elite_count,
        "target_z": args.target_z,
        "env_contract": env_contract,
        "training": {
            "best_score": best["score"],
            "history": history,
            "replay_path": rel_or_abs(replay_path),
            "replay_summary": replay_summary,
            "memory_loaded": learner_memory.get("summary", {}),
            "deterministic_eval_seeds": eval_seeds,
        },
        "budget": budget,
        "evaluation": evaluation,
        "score": evaluation["score"],
        "policy_path": rel_or_abs(policy_path),
        "transfer_artifact_path": rel_or_abs(transfer_artifact_path),
        "residuals": residuals,
        "checks": [
            check("local_training_loop_completed", True, f"iterations={args.iterations} population={args.population}"),
            check("persistent_replay_enabled", bool(replay_rows), rel_or_abs(replay_path)),
            check("state_normalization_defined", True, json.dumps(state_normalizer_spec(args.source), sort_keys=True)),
            check("action_scaling_defined", bool(policy.get("action_adapter")), json.dumps(policy.get("action_adapter"), sort_keys=True)),
            check("deterministic_eval_seeds", len(eval_seeds) >= 1, ",".join(str(seed) for seed in eval_seeds)),
            check(
                "train_before_eval_candidate_budget",
                budget["candidate_evaluations"] >= budget["min_train_candidate_evaluations"],
                f"{budget['candidate_evaluations']}>={budget['min_train_candidate_evaluations']}",
            ),
            check(
                "train_before_eval_env_step_budget",
                budget["train_env_steps"] >= budget["min_train_env_steps"],
                f"{budget['train_env_steps']}>={budget['min_train_env_steps']}",
            ),
            check("external_inference_zero", True, "numpy/local simulator only"),
            check("sim_only_no_live_hardware", True, "no MAVLink/live endpoint; gui disabled"),
            check("env_contract_probed", bool(env_contract.get("action_shape")), env_contract.get("env_id", args.source)),
            check("controller_policy_written", policy_path.exists(), rel_or_abs(policy_path)),
            check("transfer_artifact_written", transfer_artifact_path.exists(), rel_or_abs(transfer_artifact_path)),
            check(
                "rlds_minari_trace_export",
                bool(evaluation.get("trace_manifest_path")),
                str(evaluation.get("trace_manifest_path") or "trace disabled"),
            ),
        ],
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    if args.out:
        write_json(resolve_path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0


def train_controller(
    args: argparse.Namespace,
    rng: np.random.Generator,
    learner_memory: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if is_pyflyt_source(args.source):
        mean = np.array([0.50, 0.50, 0.18, 0.00, 0.08, 0.00, 0.08, 0.00, 0.03, 0.0, 0.0, 0.0], dtype=np.float64)
        std = np.array([0.12, 0.25, 0.10, 0.12, 0.08, 0.12, 0.08, 0.06, 0.04, 0.02, 0.02, 0.02], dtype=np.float64)
        seed_bank = [
            np.array([bias, 0.35, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            for bias in (0.35, 0.45, 0.55, 0.65, 0.75)
        ]
        if is_pyflyt_waypoints(args.source):
            mean = np.array([0.52, 0.65, 0.18, 0.85, 0.16, -0.85, -0.16, 0.00, 0.02, 0.0, 0.0, 0.0], dtype=np.float64)
            std = np.array([0.12, 0.32, 0.10, 0.32, 0.12, 0.32, 0.12, 0.06, 0.04, 0.03, 0.03, 0.03], dtype=np.float64)
            seed_bank = [
                np.array([0.50, 0.60, 0.14, y_gain, 0.10, x_gain, -0.10, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
                for x_gain, y_gain in ((-0.65, 0.65), (-0.95, 0.95), (0.65, -0.65), (0.95, -0.95), (0.0, 0.0))
            ]
    else:
        mean = np.array([0.25, 0.35, 0.08, 0.02, 0.01, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        std = np.array([0.18, 0.22, 0.08, 0.05, 0.03, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float64)
        seed_bank = [
            np.array([bias, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            for bias in (0.0, 0.1, 0.2, 0.3, 0.4)
        ]
    best = {"score": -1.0, "params": mean.tolist(), "metrics": {}}
    history: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    iterations = max(1, int(args.iterations))
    population = max(4, int(args.population))
    elite_count = max(1, min(int(args.elite_count), population))
    max_iterations = max_extra_iterations(args, iterations, population)
    iteration = 0
    while iteration < iterations or not training_budget_met(args, replay_rows):
        candidates: list[np.ndarray] = []
        if iteration == 0:
            candidates.extend(seed_bank)
            candidates.extend(np.asarray(row, dtype=np.float64) for row in learner_memory.get("prior_params", [])[: max(1, elite_count)])
        while len(candidates) < population:
            candidates.append(mean + rng.normal(0.0, std))
        scored = []
        for idx, params in enumerate(candidates[:population]):
            params = np.clip(params, -2.0, 2.0)
            metrics = evaluate_controller(params.tolist(), args, episodes=1, seed_offset=iteration * 100 + idx)
            scored.append({"score": metrics["score"], "params": params, "metrics": metrics})
            replay_rows.append(
                {
                    "created_utc": now(),
                    "source": args.source,
                    "seed": args.seed,
                    "iteration": iteration,
                    "candidate_index": idx,
                    "score": round(float(metrics["score"]), 8),
                    "params": [round(float(value), 8) for value in params.tolist()],
                    "metrics": replay_metric_view(metrics),
                }
            )
            if metrics["score"] > best["score"]:
                best = {"score": metrics["score"], "params": params.tolist(), "metrics": metrics}
        scored.sort(key=lambda row: row["score"], reverse=True)
        elites = np.array([row["params"] for row in scored[:elite_count]], dtype=np.float64)
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 0.015) * 0.82
        history.append(
            {
                "iteration": iteration,
                "best_score": round(float(scored[0]["score"]), 6),
                "mean_elite_score": round(float(sum(row["score"] for row in scored[:elite_count]) / elite_count), 6),
                "best_survival": scored[0]["metrics"].get("survival_ratio"),
                "best_reward_per_step": scored[0]["metrics"].get("reward_per_step"),
            }
        )
        iteration += 1
        if iteration >= max_iterations and not training_budget_met(args, replay_rows):
            break
    return best, history, replay_rows


def evaluate_controller(
    params: list[float],
    args: argparse.Namespace,
    *,
    episodes: int,
    seed_offset: int,
    seed_sequence: list[int] | None = None,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    totals = {
        "steps": 0,
        "total_reward": 0.0,
        "terminated": 0,
        "truncated": 0,
        "altitude_error_sum": 0.0,
        "xy_error_sum": 0.0,
        "vz_abs_sum": 0.0,
        "action_delta_sum": 0.0,
        "target_distance_sum": 0.0,
        "samples": 0,
    }
    final_altitudes: list[float] = []
    trace_rows = 0
    trace_file = None
    trace_manifest_path = None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path.open("w", encoding="utf-8")
        trace_manifest_path = trace_path.with_suffix(".manifest.json")
    seeds = seed_sequence or [args.seed + seed_offset + episode for episode in range(max(1, episodes))]
    for episode, episode_seed in enumerate(seeds[: max(1, episodes)]):
        env = make_env(args.source)
        obs = reset_env(env, episode_seed)
        previous_action = None
        try:
            for step_index in range(max(1, args.steps)):
                action = controller_action(params, obs, env.action_space, args.target_z, args.source)
                step = env.step(action)
                obs = step[0]
                reward = float(step[1]) if len(step) > 1 else 0.0
                terminated = bool(step[2]) if len(step) > 2 else False
                truncated = bool(step[3]) if len(step) > 3 else False
                info = step[4] if len(step) > 4 and isinstance(step[4], dict) else {}
                features = state_features(obs, args.source)
                totals["steps"] += 1
                totals["total_reward"] += reward
                target_dx = features.get("target_dx")
                target_dy = features.get("target_dy")
                target_dz = features.get("target_dz")
                if target_dx is not None and target_dy is not None and target_dz is not None:
                    altitude_error = abs(float(target_dz))
                    xy_error = math.sqrt(float(target_dx) ** 2 + float(target_dy) ** 2)
                    target_distance = math.sqrt(float(target_dx) ** 2 + float(target_dy) ** 2 + float(target_dz) ** 2)
                else:
                    altitude_error = abs(float(features["z"]) - args.target_z)
                    xy_error = math.sqrt(float(features["x"]) ** 2 + float(features["y"]) ** 2)
                    target_distance = math.sqrt(float(features["x"]) ** 2 + float(features["y"]) ** 2 + altitude_error**2)
                totals["altitude_error_sum"] += altitude_error
                totals["xy_error_sum"] += xy_error
                totals["target_distance_sum"] += target_distance
                totals["vz_abs_sum"] += abs(float(features["vz"]))
                if previous_action is not None:
                    totals["action_delta_sum"] += float(np.mean(np.abs(np.asarray(action) - np.asarray(previous_action))))
                previous_action = np.asarray(action)
                totals["samples"] += 1
                final_altitudes.append(float(features["z"]))
                if trace_file is not None:
                    trace_file.write(
                        json.dumps(
                            {
                                "episode": episode,
                                "seed": episode_seed,
                                "step": step_index,
                                "features": round_dict(features),
                                "normalized_features": round_dict(normalized_state_features(features, args.source)),
                                "action": [round(float(value), 8) for value in np.asarray(action).reshape(-1).tolist()],
                                "reward": round(reward, 8),
                                "terminated": terminated,
                                "truncated": truncated,
                                "info": scrub_info(info),
                                "obs_digest": observation_digest(obs),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    trace_rows += 1
                if terminated or truncated:
                    totals["terminated"] += int(terminated)
                    totals["truncated"] += int(truncated)
                    break
        finally:
            close_env(env)
    if trace_file is not None:
        trace_file.close()
    samples = max(1, int(totals["samples"]))
    expected_steps = max(1, episodes * max(1, args.steps))
    reward_per_step = float(totals["total_reward"]) / max(1, int(totals["steps"]))
    mean_altitude_error = float(totals["altitude_error_sum"]) / samples
    mean_xy_error = float(totals["xy_error_sum"]) / samples
    mean_abs_vz = float(totals["vz_abs_sum"]) / samples
    mean_action_delta = float(totals["action_delta_sum"]) / max(1, samples - episodes)
    mean_target_distance = float(totals["target_distance_sum"]) / samples
    survival_ratio = float(totals["steps"]) / expected_steps
    termination_rate = float(totals["terminated"] + totals["truncated"]) / max(1, episodes)
    reward_score = clamp01((reward_per_step + (1.0 if is_pyflyt_waypoints(args.source) else 0.20)) / (4.0 if is_pyflyt_waypoints(args.source) else 2.05))
    altitude_score = 1.0 - clamp01(mean_altitude_error / 1.25)
    xy_score = 1.0 - clamp01(mean_xy_error / 2.0)
    vz_score = 1.0 - clamp01(mean_abs_vz / 2.0)
    smooth_score = 1.0 - clamp01(mean_action_delta / 1.0)
    target_score = 1.0 - clamp01(mean_target_distance / 8.0)
    if is_pyflyt_waypoints(args.source):
        score = clamp01(
            0.25 * survival_ratio
            + 0.35 * target_score
            + 0.20 * reward_score
            + 0.10 * smooth_score
            + 0.05 * altitude_score
            + 0.05 * vz_score
            - 0.10 * termination_rate
        )
    else:
        score = clamp01(
            0.30 * survival_ratio
            + 0.25 * reward_score
            + 0.25 * altitude_score
            + 0.10 * vz_score
            + 0.05 * xy_score
            + 0.05 * smooth_score
            - 0.10 * termination_rate
        )
    evaluation = {
        "score": round(float(score), 8),
        "steps": int(totals["steps"]),
        "expected_steps": expected_steps,
        "total_reward": round(float(totals["total_reward"]), 8),
        "reward_per_step": round(reward_per_step, 8),
        "survival_ratio": round(survival_ratio, 8),
        "termination_rate": round(termination_rate, 8),
        "mean_altitude_error": round(mean_altitude_error, 8),
        "final_altitude": round(final_altitudes[-1], 8) if final_altitudes else None,
        "mean_xy_error": round(mean_xy_error, 8),
        "mean_target_distance": round(mean_target_distance, 8),
        "mean_abs_vz": round(mean_abs_vz, 8),
        "mean_action_delta": round(mean_action_delta, 8),
        "component_scores": {
            "reward": round(float(reward_score), 8),
            "altitude": round(float(altitude_score), 8),
            "xy": round(float(xy_score), 8),
            "target_distance": round(float(target_score), 8),
            "vertical_velocity": round(float(vz_score), 8),
            "smoothness": round(float(smooth_score), 8),
        },
    }
    if trace_path is not None and trace_manifest_path is not None:
        manifest = {
            "schema": "theseus_drone_rollout_trace_v1",
            "created_utc": now(),
            "source": args.source,
            "seed": args.seed,
            "eval_seeds": seeds[: max(1, episodes)],
            "controller": controller_name(args.source),
            "trace_path": rel_or_abs(trace_path),
            "rows": trace_rows,
            "episodes": max(1, episodes),
            "requested_steps": max(1, args.steps),
            "score": evaluation["score"],
            "external_inference_calls": 0,
            "intended_consumers": [
                "arm_transfer_policy_priors",
                "residual_escrow_replay",
                "teacher_architecture_wall_evidence",
                "rlds_minari_gateway",
            ],
        }
        write_json(trace_manifest_path, manifest)
        evaluation["trace_path"] = rel_or_abs(trace_path)
        evaluation["trace_manifest_path"] = rel_or_abs(trace_manifest_path)
        evaluation["trace_rows"] = trace_rows
    return evaluation


def make_env(source: str) -> Any:
    if source == "pyflyt":
        import gymnasium as gym
        import PyFlyt.gym_envs  # noqa: F401

        return gym.make("PyFlyt/QuadX-Hover-v4")
    if source == "pyflyt_waypoints":
        import gymnasium as gym
        import PyFlyt.gym_envs  # noqa: F401

        return gym.make("PyFlyt/QuadX-Waypoints-v4")
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary

    return HoverAviary(gui=False)


def reset_env(env: Any, seed: int) -> Any:
    out = env.reset(seed=seed)
    return out[0] if isinstance(out, tuple) else out


def close_env(env: Any) -> None:
    try:
        env.close()
    except Exception:
        pass


def state_features(obs: Any, source: str = "") -> dict[str, float]:
    if isinstance(obs, dict):
        base = state_features(obs.get("attitude", []), source or "pyflyt")
        target = np.asarray(obs.get("target_deltas", []), dtype=np.float64).reshape(-1)
        if target.size >= 3:
            base["target_dx"] = float(target[0])
            base["target_dy"] = float(target[1])
            base["target_dz"] = float(target[2])
        return base
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    def at(index: int, default: float = 0.0) -> float:
        return float(arr[index]) if index < arr.size and np.isfinite(arr[index]) else default

    if is_pyflyt_source(source) and arr.size >= 13:
        return {
            "x": at(10),
            "y": at(11),
            "z": at(12, 0.0),
            "vx": at(7),
            "vy": at(8),
            "vz": at(9),
            "roll_rate": at(0),
            "pitch_rate": at(1),
            "yaw_rate": at(2),
        }
    return {
        "x": at(0),
        "y": at(1),
        "z": at(2, 0.0),
        "vx": at(9),
        "vy": at(10),
        "vz": at(11),
    }


def state_normalizer_spec(source: str) -> dict[str, Any]:
    scales = {
        "x": state_scale(source, "x"),
        "y": state_scale(source, "y"),
        "z": state_scale(source, "z"),
        "vx": state_scale(source, "vx"),
        "vy": state_scale(source, "vy"),
        "vz": state_scale(source, "vz"),
        "target_dx": state_scale(source, "target_dx"),
        "target_dy": state_scale(source, "target_dy"),
        "target_dz": state_scale(source, "target_dz"),
        "roll_rate": state_scale(source, "roll_rate"),
        "pitch_rate": state_scale(source, "pitch_rate"),
        "yaw_rate": state_scale(source, "yaw_rate"),
    }
    return {
        "schema": "theseus_drone_state_normalizer_v1",
        "mode": "bounded_scale_clip",
        "clip": [-2.0, 2.0],
        "scales": scales,
    }


def normalized_state_features(features: dict[str, float], source: str) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in features.items():
        if not isinstance(value, (float, int)):
            continue
        scale = state_scale(source, key)
        normalized[key] = max(-2.0, min(2.0, float(value) / max(scale, 1e-6)))
    return normalized


def state_scale(source: str, key: str) -> float:
    if is_pyflyt_waypoints(source):
        table = {
            "target_dx": 4.0,
            "target_dy": 4.0,
            "target_dz": 2.0,
            "x": 4.0,
            "y": 4.0,
            "z": 2.0,
            "vx": 3.0,
            "vy": 3.0,
            "vz": 2.0,
            "roll_rate": 4.0,
            "pitch_rate": 4.0,
            "yaw_rate": 4.0,
        }
        return table.get(key, 1.0)
    if is_pyflyt_source(source):
        table = {
            "x": 2.0,
            "y": 2.0,
            "z": 2.0,
            "vx": 2.0,
            "vy": 2.0,
            "vz": 2.0,
            "roll_rate": 4.0,
            "pitch_rate": 4.0,
            "yaw_rate": 4.0,
        }
        return table.get(key, 1.0)
    return 2.0 if key in {"x", "y", "z", "vx", "vy", "vz"} else 1.0


def controller_action(params: list[float], obs: Any, action_space: Any, target_z: float, source: str) -> np.ndarray:
    p = np.asarray(params, dtype=np.float64)
    if p.size < 12:
        p = np.pad(p, (0, 12 - p.size))
    s = state_features(obs, source)
    n = normalized_state_features(s, source)
    if is_pyflyt_source(source):
        if is_pyflyt_waypoints(source) and {"target_dx", "target_dy", "target_dz"}.issubset(s):
            thrust = p[0] + p[1] * n["target_dz"] + p[2] * (-n["vz"])
            roll_rate = p[3] * n["target_dy"] + p[4] * (-n["vy"])
            pitch_rate = p[5] * n["target_dx"] + p[6] * (-n["vx"])
            yaw_rate = p[7] * (-n.get("yaw_rate", 0.0)) + p[8]
            action = np.array([roll_rate, pitch_rate, yaw_rate, thrust], dtype=np.float64)
            return clip_to_action_space(action, action_space)
        thrust = p[0] + p[1] * ((target_z - s["z"]) / state_scale(source, "z")) + p[2] * (-n["vz"])
        roll_rate = p[3] * (-n["y"]) + p[4] * (-n["vy"])
        pitch_rate = p[5] * (n["x"]) + p[6] * (n["vx"])
        yaw_rate = p[7] * (-n.get("yaw_rate", 0.0)) + p[8]
        action = np.array([roll_rate, pitch_rate, yaw_rate, thrust], dtype=np.float64)
        return clip_to_action_space(action, action_space)
    base = p[0] + p[1] * (target_z - s["z"]) + p[2] * (-s["vz"])
    roll = p[3] * (-s["x"]) + p[4] * (-s["vx"])
    pitch = p[5] * (-s["y"]) + p[6] * (-s["vy"])
    motors = np.array(
        [
            base + roll + pitch + p[8] * 0.05,
            base - roll + pitch + p[9] * 0.05,
            base + roll - pitch + p[10] * 0.05,
            base - roll - pitch + p[11] * 0.05,
        ],
        dtype=np.float32,
    )
    shape = tuple(getattr(action_space, "shape", None) or motors.shape)
    size = int(np.prod(shape)) if shape else 4
    if size != 4:
        action = np.full(size, base, dtype=np.float32)
    else:
        action = motors
    return clip_to_action_space(action, action_space).reshape(shape)


def clip_to_action_space(action: np.ndarray, action_space: Any) -> np.ndarray:
    shape = tuple(getattr(action_space, "shape", None) or action.shape)
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    low_arr = np.asarray(low, dtype=np.float64).reshape(-1) if low is not None else np.full(int(np.prod(shape)), -1.0)
    high_arr = np.asarray(high, dtype=np.float64).reshape(-1) if high is not None else np.full(int(np.prod(shape)), 1.0)
    flat = np.asarray(action, dtype=np.float64).reshape(-1)
    if flat.size != low_arr.size:
        flat = np.resize(flat, low_arr.size)
    return np.clip(flat, low_arr, high_arr).astype(np.float32).reshape(shape)


def controller_name(source: str) -> str:
    if is_pyflyt_waypoints(source):
        return "pyflyt_rate_thrust_waypoint_controller"
    return "pyflyt_rate_thrust_hover_controller" if is_pyflyt_source(source) else "linear_altitude_xy_mixer"


def probe_env_contract(source: str, seed: int) -> dict[str, Any]:
    env = make_env(source)
    try:
        obs = reset_env(env, seed)
        zero_action = clip_to_action_space(np.zeros(getattr(env.action_space, "shape", (4,))), env.action_space)
        step = env.step(zero_action)
        next_obs = step[0]
        return {
            "env_id": pyflyt_env_id(source) if is_pyflyt_source(source) else type(env).__name__,
            "source": source,
            "obs_schema": observation_schema(obs, source),
            "action_shape": list(getattr(env.action_space, "shape", []) or []),
            "action_low": finite_list(getattr(env.action_space, "low", [])),
            "action_high": finite_list(getattr(env.action_space, "high", [])),
            "zero_action_reward": round(float(step[1]) if len(step) > 1 else 0.0, 8),
            "zero_action_terminated": bool(step[2]) if len(step) > 2 else False,
            "zero_action_truncated": bool(step[3]) if len(step) > 3 else False,
            "reset_obs_digest": observation_digest(obs),
            "zero_step_obs_digest": observation_digest(next_obs),
        }
    finally:
        close_env(env)


def observation_schema(obs: Any, source: str) -> dict[str, Any]:
    if isinstance(obs, dict):
        return {
            "kind": "dict",
            "keys": sorted(str(key) for key in obs.keys()),
            "attitude": observation_schema(obs.get("attitude", []), source),
            "target_deltas_shape": list(np.asarray(obs.get("target_deltas", [])).shape),
        }
    arr = np.asarray(obs).reshape(-1)
    schema = {"kind": "array", "flat_size": int(arr.size)}
    if is_pyflyt_source(source):
        schema["field_order"] = [
            "ang_vel[0:3]",
            "quaternion[3:7]",
            "lin_vel[7:10]",
            "lin_pos[10:13]",
            "previous_action[13:17]",
            "aux[17:21]",
        ]
    return schema


def is_pyflyt_source(source: str) -> bool:
    return str(source).startswith("pyflyt")


def is_pyflyt_waypoints(source: str) -> bool:
    return str(source) == "pyflyt_waypoints"


def pyflyt_env_id(source: str) -> str:
    return "PyFlyt/QuadX-Waypoints-v4" if is_pyflyt_waypoints(source) else "PyFlyt/QuadX-Hover-v4"


def observation_digest(obs: Any) -> str:
    data = json.dumps(to_jsonable(obs), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    try:
        arr = np.asarray(value)
        if arr.ndim > 0:
            return np.round(arr.astype(float), 6).tolist()
    except Exception:
        pass
    try:
        if np.isfinite(value):
            return round(float(value), 6)
    except Exception:
        return str(value)
    return str(value)


def finite_list(value: Any) -> list[float]:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return []
    return [round(float(x), 8) for x in arr.tolist() if np.isfinite(x)]


def round_dict(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 8) for key, value in values.items() if isinstance(value, (float, int))}


def scrub_info(info: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, (bool, int, float, str)):
            clean[str(key)] = value
        else:
            clean[str(key)] = to_jsonable(value)
    return clean


def resolve_trace_path(args: argparse.Namespace) -> Path:
    if args.trace_out:
        path = resolve_path(args.trace_out)
    else:
        path = ROOT / "reports" / "drone_traces" / f"{args.source}_seed{args.seed}.jsonl"
    return path


def residuals_for_evaluation(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    residuals: list[dict[str, Any]] = []
    if float(evaluation.get("survival_ratio") or 0.0) < 1.0:
        residuals.append({"type": "survival", "detail": "controller terminates before full episode length"})
    if float(evaluation.get("mean_altitude_error") or 0.0) > 0.30:
        residuals.append({"type": "hover_altitude_error", "detail": "hover controller has not stabilized target altitude"})
    if float(evaluation.get("mean_xy_error") or 0.0) > 0.25:
        residuals.append({"type": "position_hold_error", "detail": "controller drifts from origin during hover"})
    if float(evaluation.get("mean_target_distance") or 0.0) > 1.00:
        residuals.append({"type": "waypoint_tracking_error", "detail": "controller has not reliably closed distance to waypoint targets"})
    if float(evaluation.get("mean_abs_vz") or 0.0) > 0.50:
        residuals.append({"type": "vertical_velocity_instability", "detail": "vertical velocity remains too high"})
    residuals.extend(
        [
            {"type": "waypoint_tracking_untrained", "detail": "next drone benchmark should add waypoint gates"},
            {"type": "racing_line_untrained", "detail": "AI Grand Prix lane still needs gate/racing-line reward shaping"},
            {"type": "recovery_untrained", "detail": "no perturbation recovery curriculum is active yet"},
        ]
    )
    return residuals


def load_persistent_memory(args: argparse.Namespace) -> dict[str, Any]:
    replay_path = resolve_replay_path(args)
    rows = read_jsonl_tail(replay_path, 250)
    source_rows = [row for row in rows if row.get("source") == args.source and isinstance(row.get("params"), list)]
    source_rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    prior_params = [row["params"] for row in source_rows[:12]]
    for policy in sorted((ROOT / "reports").glob(f"drone_controller_{args.source}_seed*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:8]:
        payload = read_json(policy)
        params = payload.get("params") if isinstance(payload, dict) else None
        if isinstance(params, list):
            prior_params.append(params)
    return {
        "replay_path": rel_or_abs(replay_path),
        "prior_params": prior_params[:16],
        "summary": {
            "replay_rows": len(source_rows),
            "prior_param_sets": min(len(prior_params), 16),
            "best_replay_score": round(float(source_rows[0].get("score") or 0.0), 8) if source_rows else None,
        },
    }


def resolve_replay_path(args: argparse.Namespace) -> Path:
    if args.replay_path:
        return resolve_path(args.replay_path)
    return ROOT / "reports" / "drone_replay" / f"{args.source}.jsonl"


def append_replay_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def replay_buffer_summary(path: Path, source: str) -> dict[str, Any]:
    rows = [row for row in read_jsonl_tail(path, 1000) if row.get("source") == source]
    scores = [float(row.get("score") or 0.0) for row in rows]
    return {
        "path": rel_or_abs(path),
        "rows_seen": len(rows),
        "best_score": round(max(scores), 8) if scores else None,
        "mean_recent_score": round(sum(scores[-50:]) / len(scores[-50:]), 8) if scores else None,
    }


def replay_metric_view(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "score",
        "steps",
        "expected_steps",
        "reward_per_step",
        "survival_ratio",
        "mean_target_distance",
        "mean_altitude_error",
        "mean_xy_error",
        "mean_abs_vz",
        "mean_action_delta",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def budget_accounting(args: argparse.Namespace, replay_rows: list[dict[str, Any]]) -> dict[str, int | bool]:
    train_env_steps = 0
    for row in replay_rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        train_env_steps += int(metrics.get("steps") or args.steps)
    candidate_evals = len(replay_rows)
    min_candidate_evals = max(0, int(args.min_train_candidate_evals or 0))
    min_train_env_steps = max(0, int(args.min_train_env_steps or 0))
    return {
        "candidate_evaluations": candidate_evals,
        "train_env_steps": train_env_steps,
        "requested_candidate_evaluations": max(1, int(args.iterations)) * max(4, int(args.population)),
        "requested_train_env_steps": max(1, int(args.iterations)) * max(4, int(args.population)) * max(1, int(args.steps)),
        "min_train_candidate_evaluations": min_candidate_evals,
        "min_train_env_steps": min_train_env_steps,
        "train_before_eval_sufficient": candidate_evals >= min_candidate_evals and train_env_steps >= min_train_env_steps,
    }


def training_budget_met(args: argparse.Namespace, replay_rows: list[dict[str, Any]]) -> bool:
    budget = budget_accounting(args, replay_rows)
    return bool(budget["train_before_eval_sufficient"])


def max_extra_iterations(args: argparse.Namespace, iterations: int, population: int) -> int:
    min_candidate_evals = max(0, int(args.min_train_candidate_evals or 0))
    min_train_env_steps = max(0, int(args.min_train_env_steps or 0))
    by_candidates = math.ceil(min_candidate_evals / max(1, population)) if min_candidate_evals else iterations
    by_nominal_steps = math.ceil(min_train_env_steps / max(1, population * max(1, int(args.steps)))) if min_train_env_steps else iterations
    target = max(iterations, by_candidates, by_nominal_steps)
    return max(target + 2, min(target * 3, target + 12))


def deterministic_eval_seeds(args: argparse.Namespace) -> list[int]:
    count = max(1, int(args.eval_seed_count or 1), int(args.episodes or 1))
    return [int(args.seed + 10_000 + index * 997) for index in range(count)]


def action_adapter_spec(env_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "theseus_drone_action_adapter_v1",
        "mode": "continuous_rate_thrust_clip_to_env_space",
        "shape": env_contract.get("action_shape", []),
        "low": env_contract.get("action_low", []),
        "high": env_contract.get("action_high", []),
    }


def write_drone_transfer_artifact(
    args: argparse.Namespace,
    policy_path: Path,
    policy: dict[str, Any],
    evaluation: dict[str, Any],
    residuals: list[dict[str, Any]],
) -> Path:
    if args.transfer_artifact_out:
        path = resolve_path(args.transfer_artifact_out)
    else:
        path = ROOT / "reports" / "transfer_artifacts" / f"drone_policy_prior_{args.source}.json"
    payload = {
        "schema": "project_theseus_transfer_artifact_v1",
        "artifact_id": f"drone_policy_prior_{args.source}",
        "created_utc": now(),
        "family": "drone_rl",
        "source": args.source,
        "source_arm": "drone_control_arm",
        "target_arm": "drone_control_arm",
        "loads_into": ["pressure_runner", "drone_controller_trainer", "arm_transfer_planner"],
        "policy_path": rel_or_abs(policy_path),
        "controller": policy.get("controller"),
        "score": evaluation.get("score"),
        "state_normalizer": policy.get("state_normalizer"),
        "action_adapter": policy.get("action_adapter"),
        "deterministic_eval_seeds": policy.get("deterministic_eval_seeds", []),
        "residual_curriculum": [item.get("type") for item in residuals if item.get("type")],
        "promotion_gate": "future drone frontiers must load a policy prior or produce a better one before graduation",
        "external_inference_calls": 0,
    }
    write_json(path, payload)
    return path


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": str(evidence)}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
