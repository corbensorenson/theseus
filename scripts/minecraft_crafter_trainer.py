"""Train a tiny local Crafter controller and export Theseus transfer traces.

This is deliberately modest: it learns a compact action-prior policy with CEM
inside an isolated local runtime. It does not use external inference, public
servers, launcher credentials, or commercial asset downloads.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-iterations", type=int, default=4)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--eval-seed-count", type=int, default=3)
    parser.add_argument("--trace-path", required=True)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--out", default="reports/minecraft_crafter_trainer.json")
    args = parser.parse_args()

    import crafter  # Imported inside the lane runtime.

    rng = np.random.default_rng(args.seed)
    probe_env = crafter.Env(seed=args.seed, length=max(1, args.eval_steps))
    try:
        action_names = list(getattr(probe_env, "action_names", []) or [])
        action_n = int(getattr(getattr(probe_env, "action_space", None), "n", len(action_names) or 1))
    finally:
        close_env(probe_env)
    action_names = action_names or [f"action_{idx}" for idx in range(action_n)]
    mean = np.zeros(action_n, dtype=np.float64)
    # Start biased toward movement and interaction, not no-op/idling.
    for idx, name in enumerate(action_names):
        if name in {"noop", "sleep"}:
            mean[idx] -= 0.4
        if name.startswith("move_") or name == "do":
            mean[idx] += 0.4
    std = np.ones(action_n, dtype=np.float64)

    history: list[dict[str, Any]] = []
    best_logits = mean.copy()
    best_score = -1e9
    best_eval: dict[str, Any] = {}
    train_iterations = max(1, args.train_iterations)
    population = max(4, args.population)
    elite_count = max(1, min(args.elite_count, population))

    for iteration in range(train_iterations):
        candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
        for candidate_idx in range(population):
            logits = mean + std * rng.normal(size=action_n)
            score, metrics = evaluate(crafter, logits, args.seed + iteration * 1009 + candidate_idx, max(1, args.train_steps))
            candidates.append((score, logits, metrics))
            if score > best_score:
                best_score = score
                best_logits = logits.copy()
                best_eval = metrics
        candidates.sort(key=lambda row: row[0], reverse=True)
        elites = np.stack([row[1] for row in candidates[:elite_count]], axis=0)
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 0.15)
        history.append(
            {
                "iteration": iteration,
                "best_score": float(candidates[0][0]),
                "mean_score": float(np.mean([row[0] for row in candidates])),
                "elite_score": float(np.mean([row[0] for row in candidates[:elite_count]])),
                "best_metrics": candidates[0][2],
            }
        )

    eval_rows = []
    trace_path = path_arg(args.trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as trace:
        for idx in range(max(1, args.eval_seed_count)):
            seed = args.seed + 100_000 + idx
            score, metrics = evaluate(crafter, best_logits, seed, max(1, args.eval_steps), trace, action_names)
            eval_rows.append({"seed": seed, "score": score, **metrics})

    probs = softmax(best_logits)
    avg_score = float(np.mean([row["score"] for row in eval_rows])) if eval_rows else 0.0
    avg_reward = float(np.mean([row["total_reward"] for row in eval_rows])) if eval_rows else 0.0
    avg_steps = float(np.mean([row["steps"] for row in eval_rows])) if eval_rows else 0.0
    avg_achievements = float(np.mean([row["achievement_count"] for row in eval_rows])) if eval_rows else 0.0
    normalized_score = clamp01(
        0.22
        + min(0.18, avg_steps / max(1, args.eval_steps) * 0.18)
        + min(0.24, max(0.0, avg_reward) / 5.0)
        + min(0.26, avg_achievements / 8.0)
    )

    artifact_path = path_arg(args.artifact_path)
    artifact = {
        "schema": "project_theseus_minecraft_policy_prior_v1",
        "created_utc": now(),
        "source": "crafter_cem_action_prior",
        "seed": args.seed,
        "loads_into": [
            "minecraft_world_arm",
            "video_game_play_arm",
            "minecraft_open_world_sucker",
            "crafter_bridge_sucker",
            "puffer_ocean_control_arm",
            "context_packet_memory_arm",
        ],
        "action_names": action_names,
        "action_probabilities": {name: float(prob) for name, prob in zip(action_names, probs)},
        "state_features": ["inventory", "health", "food", "drink", "energy", "achievement_events", "semantic_view"],
        "residual_curriculum": ["survival", "navigation", "resource_collection", "crafting", "combat_recovery"],
        "training": {
            "algorithm": "cem_action_prior",
            "iterations": train_iterations,
            "population": population,
            "elite_count": elite_count,
            "train_steps": args.train_steps,
            "eval_steps": args.eval_steps,
            "eval_seed_count": args.eval_seed_count,
            "candidate_evaluations": train_iterations * population,
            "train_env_steps_budget": train_iterations * population * max(1, args.train_steps),
        },
        "eval": {
            "normalized_score": normalized_score,
            "avg_raw_score": avg_score,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "avg_achievements": avg_achievements,
        },
        "trace_path": rel_or_abs(trace_path),
        "external_inference_calls": 0,
    }
    write_json(artifact_path, artifact)
    report = {
        "policy": "project_theseus_minecraft_crafter_trainer_v0",
        "created_utc": now(),
        "status": "completed",
        "score": normalized_score,
        "raw_score": avg_score,
        "best_train_score": float(best_score),
        "best_train_metrics": best_eval,
        "history": history,
        "eval": eval_rows,
        "trace_path": rel_or_abs(trace_path),
        "transfer_artifact_path": rel_or_abs(artifact_path),
        "action_probabilities": artifact["action_probabilities"],
        "external_inference_calls": 0,
    }
    out = path_arg(args.out)
    write_json(out, report)
    print(json.dumps(report, indent=2))
    return 0


def evaluate(
    crafter_module: Any,
    logits: np.ndarray,
    seed: int,
    steps: int,
    trace: Any | None = None,
    action_names: list[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    env = crafter_module.Env(seed=int(seed), length=max(1, steps))
    try:
        obs = env.reset()
        del obs
        probs = softmax(logits)
        rng = np.random.default_rng(seed)
        total_reward = 0.0
        max_achievements = 0
        last_inventory: dict[str, Any] = {}
        done = False
        step = 0
        for step in range(steps):
            action = int(rng.choice(len(probs), p=probs))
            _obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            achievements = info.get("achievements", {}) if isinstance(info, dict) else {}
            inventory = info.get("inventory", {}) if isinstance(info, dict) else {}
            max_achievements = max(max_achievements, sum(1 for value in achievements.values() if value))
            last_inventory = dict(inventory)
            if trace is not None:
                trace.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "step": step,
                            "action": action,
                            "action_name": action_names[action] if action_names and action < len(action_names) else str(action),
                            "reward": float(reward),
                            "done": bool(done),
                            "achievement_count": max_achievements,
                            "health": inventory.get("health"),
                            "food": inventory.get("food"),
                            "drink": inventory.get("drink"),
                            "energy": inventory.get("energy"),
                        }
                    )
                    + "\n"
                )
            if done:
                break
        survival = (step + 1) / max(1, steps)
        inventory_signal = sum(float(last_inventory.get(key, 0) or 0) for key in ["wood", "stone", "coal", "iron", "diamond"])
        score = float(total_reward + 0.05 * max_achievements + 0.02 * inventory_signal + 0.02 * survival)
        metrics = {
            "steps": int(step + 1),
            "total_reward": float(total_reward),
            "achievement_count": int(max_achievements),
            "inventory_signal": float(inventory_signal),
            "done": bool(done),
        }
        return score, metrics
    finally:
        close_env(env)


def close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def path_arg(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
