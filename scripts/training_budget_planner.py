"""Plan resource-aware training budgets before scoring a frontier.

The autonomy loop should not evaluate a frontier after a token smoke-sized
training attempt. This planner turns the active profile, frontier family, and
current resource reports into concrete train/eval budgets that runners can
enforce and report.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="configs/training_profiles_rtx2060super.json")
    parser.add_argument("--profile", default="inner_loop")
    parser.add_argument("--frontier-family", default="drone_rl")
    parser.add_argument("--pressure-card-id", default="")
    parser.add_argument("--mode", choices=["auto", "fixed"], default="auto")
    parser.add_argument("--out", default="reports/training_budget_plan.json")
    args = parser.parse_args()

    profiles = read_json(ROOT / args.profiles)
    report = build_budget_report(
        profiles=profiles,
        profile_name=args.profile,
        frontier_family=args.frontier_family,
        pressure_card_id=args.pressure_card_id,
        mode=args.mode,
    )
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def build_budget_report(
    *,
    profiles: dict[str, Any],
    profile_name: str,
    frontier_family: str,
    pressure_card_id: str,
    mode: str = "auto",
) -> dict[str, Any]:
    resource = read_json(ROOT / "reports" / "resource_governor.json")
    optimizer = read_json(ROOT / "reports" / "performance_optimizer.json")
    profile = (profiles.get("profiles") or {}).get(profile_name, {})
    base = dict(profile.get("pressure_runner") or {})
    scale = budget_scale(profile_name, frontier_family, resource, optimizer, mode)
    planned = scaled_pressure_budget(base, profile_name, frontier_family, pressure_card_id, scale)
    candidate_evals = int(planned["train_iterations"]) * int(planned["train_population"])
    train_env_steps = candidate_evals * int(planned["steps"])
    final_eval_steps = int(planned["eval_seed_count"]) * int(planned["steps"])
    planned.update(
        {
            "train_candidate_evaluations": candidate_evals,
            "train_env_steps_budget": train_env_steps,
            "final_eval_steps_budget": final_eval_steps,
            "min_train_candidate_evaluations": int(planned["min_train_candidate_evaluations"]),
            "min_train_env_steps": int(planned["min_train_env_steps"]),
        }
    )
    sufficient = (
        candidate_evals >= int(planned["min_train_candidate_evaluations"])
        and train_env_steps >= int(planned["min_train_env_steps"])
        and int(planned["eval_seed_count"]) >= max(1, int(planned["episodes"]))
    )
    return {
        "policy": "project_theseus_training_budget_planner_v0",
        "created_utc": now(),
        "mode": mode,
        "profile": profile_name,
        "frontier_family": frontier_family,
        "pressure_card_id": pressure_card_id,
        "base_pressure_runner": base,
        "scale": scale,
        "pressure_runner": planned,
        "checks": [
            check("train_before_eval_candidate_budget", candidate_evals >= int(planned["min_train_candidate_evaluations"]), f"{candidate_evals}>={planned['min_train_candidate_evaluations']}"),
            check("train_before_eval_env_step_budget", train_env_steps >= int(planned["min_train_env_steps"]), f"{train_env_steps}>={planned['min_train_env_steps']}"),
            check("eval_seed_budget", int(planned["eval_seed_count"]) >= max(1, int(planned["episodes"])), f"{planned['eval_seed_count']}>={planned['episodes']}"),
            check("resource_governor_can_run", bool(get_path(resource, ["decision", "can_run_requested_profile"], True)), str(get_path(resource, ["decision", "throttle_reasons"], []))),
        ],
        "summary": {
            "sufficient": sufficient,
            "train_candidate_evaluations": candidate_evals,
            "train_env_steps_budget": train_env_steps,
            "final_eval_steps_budget": final_eval_steps,
            "resource_can_run": bool(get_path(resource, ["decision", "can_run_requested_profile"], True)),
            "gpu_free_mib": get_path(resource, ["current_resources", "gpu", "memory_free_mib"], None),
            "gpu_utilization_percent": get_path(resource, ["current_resources", "gpu", "utilization_gpu_percent"], None),
            "optimizer_score": optimizer.get("score"),
        },
        "external_inference_calls": 0,
    }


def budget_scale(
    profile_name: str,
    frontier_family: str,
    resource: dict[str, Any],
    optimizer: dict[str, Any],
    mode: str,
) -> float:
    if mode == "fixed" or profile_name == "smoke":
        return 1.0
    gpu_free = number(get_path(resource, ["current_resources", "gpu", "memory_free_mib"], 0), 0.0)
    gpu_total = number(get_path(resource, ["current_resources", "gpu", "memory_total_mib"], 8192), 8192.0)
    gpu_util = number(get_path(resource, ["current_resources", "gpu", "utilization_gpu_percent"], 0), 0.0)
    can_run = bool(get_path(resource, ["decision", "can_run_requested_profile"], True))
    optimizer_score = number(optimizer.get("score"), 1.0)
    headroom = max(0.0, min(1.0, (gpu_free - 1024.0) / max(1.0, gpu_total)))
    idle_bonus = 0.35 if gpu_util < 30.0 else 0.0
    frontier_bonus = 0.25 if frontier_family in {"drone_rl", "minecraft_rl", "rl_local", "emulator_rl"} else 0.10
    profile_bonus = {"inner_loop": 0.35, "candidate": 0.60, "seed_sweep": 0.45}.get(profile_name, 0.0)
    if not can_run:
        return 1.0
    return round(max(1.0, min(2.25, 1.0 + headroom + idle_bonus + frontier_bonus + profile_bonus + max(0.0, optimizer_score - 0.8))), 3)


def scaled_pressure_budget(
    base: dict[str, Any],
    profile_name: str,
    frontier_family: str,
    pressure_card_id: str,
    scale: float,
) -> dict[str, int | float | str]:
    minimums = minimum_budget(profile_name, frontier_family, pressure_card_id)
    episodes = max(int(base.get("episodes", 2)), minimums["episodes"])
    steps = max(int(base.get("steps", 96)), minimums["steps"])
    iterations = max(int(round(int(base.get("train_iterations", 4)) * scale)), minimums["train_iterations"])
    population = max(int(round(int(base.get("train_population", 12)) * scale)), minimums["train_population"])
    elite = max(int(round(int(base.get("elite_count", 4)) * min(scale, 1.5))), minimums["elite_count"])
    eval_seed_count = max(int(base.get("eval_seed_count", episodes)), episodes, minimums["eval_seed_count"])
    min_candidate_evals = max(iterations * population, minimums["min_train_candidate_evaluations"])
    min_train_env_steps = max(min_candidate_evals * steps, minimums["min_train_env_steps"])
    return {
        "episodes": episodes,
        "steps": steps,
        "train_iterations": iterations,
        "train_population": population,
        "elite_count": min(elite, population),
        "eval_seed_count": eval_seed_count,
        "min_train_candidate_evaluations": min_candidate_evals,
        "min_train_env_steps": min_train_env_steps,
        "budget_source": "resource_aware_auto_scale",
    }


def minimum_budget(profile_name: str, frontier_family: str, pressure_card_id: str) -> dict[str, int]:
    if profile_name == "smoke":
        return {
            "episodes": 1,
            "steps": 64,
            "train_iterations": 2,
            "train_population": 8,
            "elite_count": 2,
            "eval_seed_count": 1,
            "min_train_candidate_evaluations": 16,
            "min_train_env_steps": 1024,
        }
    if frontier_family == "drone_rl" and "waypoints" in pressure_card_id:
        if profile_name == "candidate":
            return {
                "episodes": 8,
                "steps": 512,
                "train_iterations": 24,
                "train_population": 48,
                "elite_count": 10,
                "eval_seed_count": 8,
                "min_train_candidate_evaluations": 1152,
                "min_train_env_steps": 589824,
            }
        return {
            "episodes": 4,
            "steps": 256,
            "train_iterations": 10,
            "train_population": 32,
            "elite_count": 8,
            "eval_seed_count": 4,
            "min_train_candidate_evaluations": 320,
            "min_train_env_steps": 81920,
        }
    if frontier_family == "minecraft_rl":
        return {
            "episodes": 4,
            "steps": 384,
            "train_iterations": 10,
            "train_population": 28,
            "elite_count": 6,
            "eval_seed_count": 4,
            "min_train_candidate_evaluations": 280,
            "min_train_env_steps": 107520,
        }
    if frontier_family in {"drone_rl", "rl_local", "emulator_rl"}:
        return {
            "episodes": 4,
            "steps": 256,
            "train_iterations": 8,
            "train_population": 24,
            "elite_count": 6,
            "eval_seed_count": 4,
            "min_train_candidate_evaluations": 192,
            "min_train_env_steps": 49152,
        }
    return {
        "episodes": 2,
        "steps": 128,
        "train_iterations": 4,
        "train_population": 12,
        "elite_count": 4,
        "eval_seed_count": 2,
        "min_train_candidate_evaluations": 48,
        "min_train_env_steps": 6144,
    }


def check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
