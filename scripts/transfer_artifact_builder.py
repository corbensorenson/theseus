"""Materialize arm-transfer plans into loadable local artifacts.

The planner says what can transfer. This builder turns those edges into small
JSON artifacts that arms and pressure runners can load: policy priors,
verification templates, residual curricula, and reusable adapter schemas.
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
    parser.add_argument("--plan", default="reports/arm_transfer_plan.json")
    parser.add_argument("--artifact-dir", default="reports/transfer_artifacts")
    parser.add_argument("--out", default="reports/arm_transfer_artifacts.json")
    args = parser.parse_args()

    plan = read_json(ROOT / args.plan)
    artifact_dir = ROOT / args.artifact_dir
    artifacts = build_artifacts(plan, artifact_dir)
    report = {
        "policy": "project_theseus_transfer_artifact_builder_v0",
        "created_utc": now(),
        "plan": args.plan,
        "artifact_dir": args.artifact_dir,
        "summary": {
            "artifacts": len(artifacts),
            "frontier_family": get_path(plan, ["summary", "frontier_family"], None),
            "ready_edges": get_path(plan, ["summary", "ready_edges"], 0),
        },
        "artifacts": artifacts,
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def build_artifacts(plan: dict[str, Any], artifact_dir: Path) -> list[dict[str, Any]]:
    family = str(get_path(plan, ["summary", "frontier_family"], "general") or "general")
    edges = plan.get("transfer_plan") if isinstance(plan.get("transfer_plan"), list) else []
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict) or edge.get("status") != "ready":
            continue
        if family == "drone_rl":
            payload = drone_transfer_artifact(edge, index)
        elif family == "minecraft_rl":
            payload = minecraft_transfer_artifact(edge, index)
        elif family in {"coding_local_sandbox", "web_agent_local", "transfer_eval"}:
            payload = adapter_transfer_artifact(edge, family, index)
        else:
            payload = generic_transfer_artifact(edge, family, index)
        path = artifact_dir / f"{payload['artifact_id']}.json"
        write_json(path, payload)
        rows.append(artifact_summary(payload, family, path))
    if not rows:
        payload = generic_transfer_artifact({}, family, 0)
        path = artifact_dir / f"{payload['artifact_id']}.json"
        write_json(path, payload)
        rows.append(artifact_summary(payload, family, path))
    return rows


def artifact_summary(payload: dict[str, Any], family: str, path: Path) -> dict[str, Any]:
    return {
        "artifact_id": payload["artifact_id"],
        "family": family,
        "path": rel(path),
        "source_arm": payload.get("source_arm"),
        "target_arm": payload.get("target_arm"),
        "source_sucker": payload.get("source_sucker"),
        "target_sucker": payload.get("target_sucker"),
        "parent_arm": payload.get("parent_arm"),
        "sucker_chain": payload.get("sucker_chain", []),
        "loads_into": payload.get("loads_into", []),
        "verification": payload.get("verification", []),
    }


def drone_transfer_artifact(edge: dict[str, Any], index: int) -> dict[str, Any]:
    latest_gym = latest_report("drone_controller_gym_pybullet_drones_seed*_trainer.json")
    latest_pyflyt = latest_report("drone_controller_pyflyt_seed*_trainer.json")
    latest_waypoints = latest_report("drone_controller_pyflyt_waypoints_seed*_trainer.json")
    latest_prior = latest_report("transfer_artifacts/drone_policy_prior_pyflyt_waypoints.json")
    return {
        "schema": "project_theseus_transfer_artifact_v0",
        "artifact_id": f"drone_control_transfer_{index}",
        "created_utc": now(),
        "source_arm": edge.get("source_arm"),
        "target_arm": edge.get("target_arm"),
        "source_sucker": edge.get("source_sucker"),
        "target_sucker": edge.get("target_sucker"),
        "parent_arm": edge.get("parent_arm") or edge.get("target_arm"),
        "sucker_chain": edge.get("sucker_chain", []),
        "loads_into": [item for item in [
            "drone_control_arm",
            "drone_racing_control_arm",
            edge.get("target_sucker"),
            "pressure_runner",
            "architecture_experiment_runner",
        ] if item],
        "transferable_structure": edge.get("transferable_structure", []),
        "controller_priors": {
            "control_form": "linear_altitude_xy_mixer",
            "hover_altitude_target": get_path(latest_gym, ["evaluation", "altitude_target"], None),
            "gym_policy_path": latest_gym.get("policy_path"),
            "pyflyt_policy_path": latest_pyflyt.get("policy_path"),
            "pyflyt_waypoints_policy_path": latest_waypoints.get("policy_path"),
            "pyflyt_waypoints_prior_path": latest_prior.get("policy_path") or "reports/transfer_artifacts/drone_policy_prior_pyflyt_waypoints.json",
            "state_normalizer": get_path(latest_prior, ["state_normalizer"], {}),
            "action_adapter": get_path(latest_prior, ["action_adapter"], {}),
            "safety": "simulator_only_until_human_approval",
        },
        "residual_curriculum": [
            "hover_altitude_error",
            "waypoint_tracking",
            "racing_line_following",
            "recovery_after_disturbance",
            "obstacle_clearance",
        ],
        "verification": edge.get("verification", []),
        "promotion_gate": "target frontier score improves or residual cluster narrows with external_inference_calls=0",
        "external_inference_calls": 0,
    }


def minecraft_transfer_artifact(edge: dict[str, Any], index: int) -> dict[str, Any]:
    latest_pressure = latest_report("pressure_source_crafter_seed*.json") or latest_report("pressure_source_craftax_seed*.json")
    latest_prior = latest_report("transfer_artifacts/minecraft_policy_prior_*.json")
    return {
        "schema": "project_theseus_transfer_artifact_v0",
        "artifact_id": f"minecraft_world_transfer_{index}",
        "created_utc": now(),
        "source_arm": edge.get("source_arm"),
        "target_arm": edge.get("target_arm"),
        "source_sucker": edge.get("source_sucker"),
        "target_sucker": edge.get("target_sucker"),
        "parent_arm": edge.get("parent_arm") or edge.get("target_arm"),
        "sucker_chain": edge.get("sucker_chain", []),
        "loads_into": [item for item in [
            "minecraft_world_arm",
            "video_game_play_arm",
            edge.get("target_sucker"),
            "pressure_runner",
            "context_packet_memory_arm",
            "rl_control_arm",
            "puffer_ocean_control_arm",
        ] if item],
        "transferable_structure": edge.get("transferable_structure", []),
        "policy_priors": {
            "bridge": "crafter_or_craftax",
            "parent_arm": edge.get("parent_arm") or edge.get("target_arm"),
            "target_sucker": edge.get("target_sucker"),
            "latest_pressure_score": get_path(latest_pressure, ["summary", "accuracy"], None),
            "latest_trace_path": get_path(latest_pressure, ["metrics", "trace_path"], latest_prior.get("trace_path")),
            "state_features": latest_prior.get(
                "state_features",
                ["inventory", "health", "position_proxy", "local_view", "achievement_events"],
            ),
            "action_schema": latest_prior.get("action_schema", "minecraft_skill_prior"),
            "safety": "local_disposable_worlds_no_public_server_by_default",
        },
        "residual_curriculum": [
            "survival",
            "resource_collection",
            "navigation",
            "crafting",
            "player_instruction_following",
            "recovery_after_damage_or_lost_goal",
        ],
        "verification": edge.get("verification", []),
        "promotion_gate": "target Minecraft/Open-World score improves or residual cluster narrows with external_inference_calls=0",
        "external_inference_calls": 0,
    }


def adapter_transfer_artifact(edge: dict[str, Any], family: str, index: int) -> dict[str, Any]:
    return {
        "schema": "project_theseus_transfer_artifact_v0",
        "artifact_id": f"{family}_adapter_transfer_{index}",
        "created_utc": now(),
        "source_arm": edge.get("source_arm"),
        "target_arm": edge.get("target_arm"),
        "source_sucker": edge.get("source_sucker"),
        "target_sucker": edge.get("target_sucker"),
        "parent_arm": edge.get("parent_arm") or edge.get("target_arm"),
        "sucker_chain": edge.get("sucker_chain", []),
        "loads_into": ["benchmark_adapter_factory", "pressure_runner"],
        "adapter_template": {
            "required_outputs": ["summary.accuracy", "checks", "residuals", "external_inference_calls"],
            "required_gates": ["license_allowed", "smoke_passed", "no_network_during_scoring"],
            "residual_types": ["runtime_blocked", "solver_absent", "service_setup", "scorer_missing"],
        },
        "transferable_structure": edge.get("transferable_structure", []),
        "verification": edge.get("verification", []),
        "external_inference_calls": 0,
    }


def generic_transfer_artifact(edge: dict[str, Any], family: str, index: int) -> dict[str, Any]:
    return {
        "schema": "project_theseus_transfer_artifact_v0",
        "artifact_id": f"{family}_generic_transfer_{index}",
        "created_utc": now(),
        "source_arm": edge.get("source_arm", "benchmark_ratchet_arm"),
        "target_arm": edge.get("target_arm", "active_frontier_arm"),
        "source_sucker": edge.get("source_sucker"),
        "target_sucker": edge.get("target_sucker"),
        "parent_arm": edge.get("parent_arm") or edge.get("target_arm", "active_frontier_arm"),
        "sucker_chain": edge.get("sucker_chain", []),
        "loads_into": ["autonomy_cycle", "arm_transfer_planner"],
        "transferable_structure": edge.get(
            "transferable_structure",
            ["threshold decay", "residual escrow", "regression promotion"],
        ),
        "verification": edge.get(
            "verification",
            ["source regression remains green", "target residual narrows", "external inference calls remain zero"],
        ),
        "external_inference_calls": 0,
    }


def latest_report(pattern: str) -> dict[str, Any]:
    reports = sorted((ROOT / "reports").glob(pattern), key=lambda item: item.stat().st_mtime)
    return read_json(reports[-1]) if reports else {}


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


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
