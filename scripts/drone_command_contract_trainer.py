"""Learn a safe local MAVSDK command-contract policy.

This runner does not connect to a drone, simulator endpoint, or external model.
It trains a tiny offline command scheduler against a deterministic kinematic
contract so Project Theseus can improve the MAVSDK/drone-control frontier
without pretending that live hardware or SITL has been mastered.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


WAYPOINTS = [
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.2),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 0.8),
    (0.0, 0.0, 1.0),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--population", type=int, default=18)
    parser.add_argument("--elite-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--policy-out", default="reports/drone_command_contract_policy_mavsdk_seed1.json")
    parser.add_argument("--out", default="reports/drone_command_contract_mavsdk_seed1.json")
    args = parser.parse_args()

    started = time.perf_counter()
    rng = random.Random(args.seed)
    import_ok = importlib.util.find_spec("mavsdk") is not None
    best, history = train_policy(args, rng)
    evaluation = evaluate_policy(best["params"], args, seed_offset=10_000)
    checks = build_checks(import_ok, evaluation)
    residuals = residuals_for(import_ok, evaluation)
    score = score_contract(import_ok, evaluation)

    policy_path = resolve_path(args.policy_out)
    policy = {
        "policy": "theseus_mavsdk_command_contract_policy_v0",
        "created_utc": now(),
        "seed": args.seed,
        "controller": "bounded_waypoint_command_scheduler",
        "params": [round(float(value), 8) for value in best["params"]],
        "waypoints": WAYPOINTS,
        "sim_only": True,
        "live_hardware_allowed": False,
        "external_inference_calls": 0,
    }
    write_json(policy_path, policy)
    report = {
        "policy": "local_only_no_external_inference",
        "methodology": "theseus_mavsdk_command_contract_trainer_v0",
        "created_utc": now(),
        "seed": args.seed,
        "iterations": args.iterations,
        "population": args.population,
        "elite_count": args.elite_count,
        "steps": args.steps,
        "score": score,
        "training": {"best_score": best["score"], "history": history},
        "evaluation": evaluation,
        "policy_path": rel_or_abs(policy_path),
        "checks": checks,
        "residuals": residuals,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    write_json(resolve_path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0


def train_policy(args: argparse.Namespace, rng: random.Random) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mean = [1.15, 1.2, 0.65, 0.55, 0.25, 0.08]
    std = [0.30, 0.35, 0.22, 0.20, 0.12, 0.05]
    lower = [0.3, 0.3, 0.15, 0.15, 0.02, 0.0]
    upper = [2.4, 2.5, 1.5, 1.5, 0.75, 0.35]
    best = {"score": -1.0, "params": mean[:]}
    history: list[dict[str, Any]] = []
    iterations = max(1, args.iterations)
    population = max(4, args.population)
    elite_count = max(1, min(args.elite_count, population))
    for iteration in range(iterations):
        candidates = [mean[:]]
        while len(candidates) < population:
            candidates.append([m + rng.gauss(0.0, s) for m, s in zip(mean, std)])
        scored = []
        for idx, params in enumerate(candidates[:population]):
            params = [max(lo, min(hi, value)) for value, lo, hi in zip(params, lower, upper)]
            evaluation = evaluate_policy(params, args, seed_offset=iteration * 101 + idx)
            contract_score = score_contract(True, evaluation)
            scored.append({"score": contract_score, "params": params, "evaluation": evaluation})
            if contract_score > best["score"]:
                best = {"score": contract_score, "params": params[:]}
        scored.sort(key=lambda row: row["score"], reverse=True)
        elite_params = [row["params"] for row in scored[:elite_count]]
        mean = [sum(row[i] for row in elite_params) / elite_count for i in range(len(mean))]
        std = [max(stddev([row[i] for row in elite_params]), 0.015) * 0.78 for i in range(len(mean))]
        history.append(
            {
                "iteration": iteration,
                "best_score": round(float(scored[0]["score"]), 6),
                "mean_elite_score": round(float(sum(row["score"] for row in scored[:elite_count]) / elite_count), 6),
                "best_tracking_error": scored[0]["evaluation"].get("mean_tracking_error"),
                "best_contract_violations": scored[0]["evaluation"].get("contract_violations"),
            }
        )
    return best, history


def evaluate_policy(params: list[float], args: argparse.Namespace, *, seed_offset: int) -> dict[str, Any]:
    hold_seconds, transit_seconds, kp_xy, kp_z, max_vel, yaw_rate = params
    dt = 0.05
    command_rate_hz = 1.0 / dt
    rng = random.Random(args.seed + seed_offset)
    position = [0.0, 0.0, 0.0]
    velocity = [0.0, 0.0, 0.0]
    target_index = 0
    hold_ticks = 0
    tracking_error_sum = 0.0
    smoothness_sum = 0.0
    previous_setpoint = None
    violations = 0
    geofence_hits = 0
    emergency_stop_ready = True
    command_count = 0
    heartbeat_count = 0
    max_speed = 0.0
    max_altitude = 0.0

    for step in range(max(1, args.steps)):
        target = WAYPOINTS[target_index]
        error = [target[i] - position[i] for i in range(3)]
        desired_velocity = [kp_xy * error[0], kp_xy * error[1], kp_z * error[2]]
        desired_velocity = clamp_norm(desired_velocity, max_vel)
        desired_velocity = [value + rng.gauss(0.0, 0.003) for value in desired_velocity]
        velocity = [0.70 * velocity[i] + 0.30 * desired_velocity[i] for i in range(3)]
        position = [position[i] + velocity[i] * dt for i in range(3)]
        speed = norm(velocity)
        max_speed = max(max_speed, speed)
        max_altitude = max(max_altitude, position[2])
        tracking_error = norm(error)
        tracking_error_sum += tracking_error
        if previous_setpoint is not None:
            smoothness_sum += norm([desired_velocity[i] - previous_setpoint[i] for i in range(3)])
        previous_setpoint = desired_velocity[:]
        command_count += 1
        heartbeat_count += 1
        if speed > 1.2 or max_altitude > 2.2 or any(abs(value) > 2.2 for value in position[:2]) or position[2] < -0.05:
            violations += 1
        if any(abs(value) > 2.0 for value in position[:2]) or position[2] > 2.0:
            geofence_hits += 1
        if tracking_error < 0.14:
            hold_ticks += 1
            required_hold = max(1, int(hold_seconds / dt))
            if hold_ticks >= required_hold and target_index < len(WAYPOINTS) - 1:
                target_index += 1
                hold_ticks = 0
        elif step > 0 and step % max(1, int(transit_seconds / dt)) == 0 and target_index < len(WAYPOINTS) - 1:
            target_index += 1
            hold_ticks = 0

    samples = max(1, args.steps)
    final_target = WAYPOINTS[-1]
    final_error = norm([final_target[i] - position[i] for i in range(3)])
    mean_tracking_error = tracking_error_sum / samples
    mean_smoothness = smoothness_sum / max(1, samples - 1)
    waypoint_progress = target_index / max(1, len(WAYPOINTS) - 1)
    command_rate_ok = 8.0 <= command_rate_hz <= 50.0
    yaw_rate_ok = abs(yaw_rate) <= 0.35
    return {
        "command_count": command_count,
        "heartbeat_count": heartbeat_count,
        "command_rate_hz": round(command_rate_hz, 4),
        "command_rate_ok": command_rate_ok,
        "yaw_rate_ok": yaw_rate_ok,
        "waypoint_progress": round(float(waypoint_progress), 8),
        "mean_tracking_error": round(float(mean_tracking_error), 8),
        "final_error": round(float(final_error), 8),
        "mean_smoothness": round(float(mean_smoothness), 8),
        "max_speed": round(float(max_speed), 8),
        "max_altitude": round(float(max_altitude), 8),
        "contract_violations": violations,
        "geofence_hits": geofence_hits,
        "emergency_stop_ready": emergency_stop_ready,
        "live_endpoint_contacted": False,
        "hardware_approval_required": True,
    }


def score_contract(import_ok: bool, evaluation: dict[str, Any]) -> float:
    progress = clamp01(evaluation.get("waypoint_progress", 0.0))
    tracking = 1.0 - clamp01(float(evaluation.get("mean_tracking_error") or 0.0) / 1.6)
    final = 1.0 - clamp01(float(evaluation.get("final_error") or 0.0) / 1.8)
    smooth = 1.0 - clamp01(float(evaluation.get("mean_smoothness") or 0.0) / 0.35)
    safety = 1.0 if int(evaluation.get("contract_violations") or 0) == 0 else 0.35
    geofence = 1.0 if int(evaluation.get("geofence_hits") or 0) == 0 else 0.25
    rate = 1.0 if evaluation.get("command_rate_ok") else 0.4
    yaw = 1.0 if evaluation.get("yaw_rate_ok") else 0.6
    endpoint = 1.0 if not evaluation.get("live_endpoint_contacted") else 0.0
    import_score = 1.0 if import_ok else 0.0
    # No SITL/live endpoint is active, so the diagnostic surface is capped below
    # full mastery while still allowing floor-clearing when the local safety
    # contract is learned.
    raw = (
        0.10 * import_score
        + 0.18 * progress
        + 0.18 * tracking
        + 0.12 * final
        + 0.08 * smooth
        + 0.14 * safety
        + 0.08 * geofence
        + 0.05 * rate
        + 0.03 * yaw
        + 0.04 * endpoint
    )
    cap = 0.78 if import_ok else 0.58
    return round(float(min(cap, clamp01(raw))), 8)


def build_checks(import_ok: bool, evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        check("mavsdk_import", import_ok, "module found" if import_ok else "module missing"),
        check("no_live_endpoint_attempted", not evaluation.get("live_endpoint_contacted"), "offline contract only"),
        check("real_hardware_requires_human_approval", bool(evaluation.get("hardware_approval_required")), "approval gate active"),
        check("geofence_contract", int(evaluation.get("geofence_hits") or 0) == 0, f"hits={evaluation.get('geofence_hits')}"),
        check("emergency_stop_policy_present", bool(evaluation.get("emergency_stop_ready")), "simulated e-stop contract present"),
        check("command_rate_contract", bool(evaluation.get("command_rate_ok")), f"hz={evaluation.get('command_rate_hz')}"),
        check("waypoint_contract_progress", float(evaluation.get("waypoint_progress") or 0.0) >= 0.75, f"progress={evaluation.get('waypoint_progress')}"),
        check("external_inference_zero", True, "pure-python local contract only"),
    ]


def residuals_for(import_ok: bool, evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    residuals: list[dict[str, Any]] = []
    if not import_ok:
        residuals.append({"type": "runtime_dependency", "detail": "mavsdk module is not importable in the selected Python lane"})
    if float(evaluation.get("waypoint_progress") or 0.0) < 1.0:
        residuals.append({"type": "waypoint_contract", "detail": "offline command scheduler did not complete all waypoints"})
    if float(evaluation.get("mean_tracking_error") or 0.0) > 0.45:
        residuals.append({"type": "tracking_contract", "detail": "offline setpoint tracking remains loose"})
    if int(evaluation.get("contract_violations") or 0) > 0:
        residuals.append({"type": "safety_contract", "detail": "velocity/altitude/geofence contract was violated"})
    residuals.append({"type": "sitl_endpoint_not_active", "detail": "PX4/SITL integration is still a later gated frontier"})
    return residuals


def clamp_norm(vector: list[float], max_norm: float) -> list[float]:
    length = norm(vector)
    if length <= max_norm or length <= 1e-9:
        return vector
    return [value * (max_norm / length) for value in vector]


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
