"""Plan transfer-learning moves across Octopus arms.

The local model should not learn every benchmark from scratch. This planner
links source arms/tools/checkpoints to the active frontier family and emits a
machine-readable transfer plan for the autonomy loop, teacher evidence, and
dashboard.
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
    parser.add_argument("--out", default="reports/arm_transfer_plan.json")
    parser.add_argument("--markdown-out", default="reports/arm_transfer_plan.md")
    args = parser.parse_args()

    state = load_state()
    frontier = active_frontier(state)
    plan = build_plan(state, frontier)
    report = {
        "policy": "project_theseus_arm_transfer_planner_v0",
        "created_utc": now(),
        "active_frontier": frontier,
        "summary": {
            "frontier_family": frontier_family(frontier),
            "frontier_source": frontier.get("frontier_source", "benchmark_ledger"),
            "pressure_card_id": frontier.get("pressure_card_id"),
            "transfer_edges": len(plan),
            "ready_edges": sum(1 for item in plan if item.get("status") == "ready"),
            "blocked_edges": sum(1 for item in plan if item.get("status") == "blocked"),
            "sucker_edges": sum(1 for item in plan if item.get("target_sucker")),
            "target_suckers": sorted({str(item.get("target_sucker")) for item in plan if item.get("target_sucker")}),
            "arm_sucker_ready": get_path(state, ["arm_sucker", "summary", "ready_for_transfer_routing"], None),
        },
        "transfer_plan": plan,
        "next_actions": next_actions(plan),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    write_markdown(ROOT / args.markdown_out, report)
    print(json.dumps(report, indent=2))
    return 0


def load_state() -> dict[str, Any]:
    reports = ROOT / "reports"
    return {
        "frontier_policy": read_json(reports / "frontier_policy_status.json"),
        "benchmaxx_curriculum": read_json(reports / "benchmaxx_curriculum.json"),
        "benchmark_ledger": read_json(reports / "benchmark_ledger.json"),
        "arm_registry": read_json(reports / "arm_registry.json"),
        "arm_sucker": read_json(reports / "arm_sucker_registry.json"),
        "tool_registry": read_json(reports / "tool_registry.json"),
        "residual_escrow": read_json(reports / "residual_escrow.json"),
        "candidate": read_json(reports / "candidate_promotion_gate.json"),
        "drone_pressure": latest_drone_pressure(),
        "drone_traces": latest_trace_manifest(),
        "drone_policy": latest_report("drone_controller_*_trainer.json"),
        "minecraft_pressure": latest_minecraft_pressure(),
        "minecraft_runtime": read_json(reports / "minecraft_runtime_probe.json"),
        "minecraft_artifact": latest_report("transfer_artifacts/minecraft_policy_prior_*.json"),
        "transfer_eval": read_json(reports / "transfer_eval_suite.json"),
    }


def build_plan(state: dict[str, Any], frontier: dict[str, Any]) -> list[dict[str, Any]]:
    family = frontier_family(frontier)
    rows: list[dict[str, Any]] = []
    if family == "drone_rl":
        target_sucker = drone_sucker_for_frontier(frontier)
        rows.extend(
            [
                edge(
                    "puffer_ocean_control_arm",
                    "drone_racing_control_arm",
                    "Transfer survival/reward controller search from local Ocean/Puffer control to drone hover pressure.",
                    "ready" if state.get("drone_pressure") else "blocked",
                    ["policy search", "reward normalization", "residual escrow", "rollout traces"],
                    "Run pressure_runner with learned drone controller and compare score over seeds.",
                    artifact_hints(state, ["drone_pressure", "drone_policy", "drone_traces"]),
                    target_sucker=target_sucker,
                    sucker_chain=["drone_racing_control_arm", target_sucker],
                ),
                edge(
                    "safety_reflex_arm",
                    "drone_racing_control_arm",
                    "Transfer hard permission boundaries into simulator-first drone control before SITL or hardware.",
                    "ready",
                    ["sim-only", "hardware approval gate", "emergency hold semantics"],
                    "Keep MAVSDK/live endpoints contract-only until explicit human approval.",
                    target_sucker=target_sucker,
                    sucker_chain=["drone_racing_control_arm", target_sucker],
                ),
                edge(
                    "benchmark_ratchet_arm",
                    "drone_racing_control_arm",
                    "Transfer benchmark lifecycle and threshold decay to hover, waypoint, racing, and recovery curricula.",
                    "ready",
                    ["graduation floor", "residual clusters", "regression promotion"],
                    "Escrow hover failures and stage waypoint/racing bridge benchmarks.",
                    artifact_hints(state, ["drone_pressure", "drone_traces"]),
                    target_sucker=target_sucker,
                    sucker_chain=["drone_racing_control_arm", target_sucker],
                ),
            ]
        )
    elif family == "minecraft_rl":
        target_sucker = minecraft_sucker_for_frontier(frontier)
        rows.extend(
            [
                edge(
                    "puffer_ocean_control_arm",
                    "video_game_play_arm",
                    "Transfer survival/reward search and local controller pressure into Minecraft-like open-world tasks.",
                    "ready" if state.get("minecraft_pressure") else "blocked",
                    ["reward normalization", "state-action-reward-done traces", "survival/crafting residuals"],
                    "Run minecraft_rl pressure on Crafter/Craftax bridge or full local harness and export trace capsules.",
                    artifact_hints(state, ["minecraft_pressure", "minecraft_artifact"]),
                    target_sucker=target_sucker,
                    sucker_chain=["video_game_play_arm", "minecraft_open_world_sucker", target_sucker],
                ),
                edge(
                    "context_packet_memory_arm",
                    "video_game_play_arm",
                    "Transfer packet-scored long-context memory into inventory, waypoint, and player-instruction state.",
                    "ready",
                    ["inventory packets", "goal packets", "world event summaries", "importance-scored trace compaction"],
                    "Persist Minecraft trace packets and test recovery after long episodes.",
                    artifact_hints(state, ["minecraft_pressure"]),
                    target_sucker=target_sucker,
                    sucker_chain=["video_game_play_arm", "minecraft_open_world_sucker", target_sucker],
                ),
                edge(
                    "safety_reflex_arm",
                    "video_game_play_arm",
                    "Transfer account, network, and world-mutation gates into local Minecraft control.",
                    "ready",
                    ["no public server by default", "no credential storage", "disposable worlds", "interruptible player co-op"],
                    "Keep full Minecraft runtime local and approval-gated; use bridge worlds for autonomous pressure.",
                    target_sucker="minecraft_java_local_sucker",
                    sucker_chain=["video_game_play_arm", "minecraft_open_world_sucker", "minecraft_java_local_sucker"],
                ),
            ]
        )
    elif family in {"coding_local_sandbox", "web_agent_local", "transfer_eval"}:
        rows.append(
            edge(
                "code_repair_verifier",
                "benchmark_adapter_factory",
                "Transfer local unit-test/smoke verification into new benchmark adapters.",
                "ready",
                ["sandbox scoring", "adapter smoke", "residual clustering"],
                "Promote repeated adapter setup into a verified tool.",
            )
        )
    else:
        rows.append(
            edge(
                "benchmark_ratchet_arm",
                "active_frontier_arm",
                "Transfer generic ratchet mechanics into the active frontier family.",
                "ready",
                ["threshold decay", "residual escrow", "teacher escalation"],
                "Keep active frontier pressure until mastery, floor graduation, or architecture wall.",
            )
        )
    return rows


def edge(
    source_arm: str,
    target_arm: str,
    hypothesis: str,
    status: str,
    transferable_structure: list[str],
    next_action: str,
    artifacts: dict[str, Any] | None = None,
    *,
    source_sucker: str = "",
    target_sucker: str = "",
    sucker_chain: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "source_arm": source_arm,
        "target_arm": target_arm,
        "source_sucker": source_sucker or None,
        "target_sucker": target_sucker or None,
        "parent_arm": target_arm,
        "sucker_chain": [item for item in (sucker_chain or []) if item],
        "hypothesis": hypothesis,
        "status": status,
        "transferable_structure": transferable_structure,
        "verification": [
            "source regression remains green",
            "target frontier score improves or residual narrows",
            "external inference calls remain zero",
        ],
        "next_action": next_action,
        "artifacts": artifacts or {},
    }


def active_frontier(state: dict[str, Any]) -> dict[str, Any]:
    """Follow the active frontier selected by frontier_policy_status.

    Transfer can become actively harmful if it optimizes for the largest
    residual in the ledger while the autonomy loop is pressurizing a different
    benchmark family. The policy report is therefore the source of truth, with
    the ledger used only to enrich the selected row.
    """
    ledger = state.get("benchmark_ledger")
    frontiers = [row for row in ledger if isinstance(row, dict) and row.get("lifecycle") == "frontier"] if isinstance(ledger, list) else []
    preferred_family, pressure_card_id, source = effective_frontier_selection(state)
    if frontiers and (preferred_family or pressure_card_id):
        matches = [
            row
            for row in frontiers
            if policy_matches_frontier(row, preferred_family, pressure_card_id)
        ]
        if matches:
            selected = max(matches, key=lambda row: float(row.get("residual") or 0.0))
            return {
                **selected,
                "frontier_source": source,
                "pressure_card_id": pressure_card_id or None,
            }
    if pressure_card_id:
        return {
            "benchmark_name": f"{preferred_family or 'pressure'}_{pressure_card_id}",
            "benchmark_type": f"frontier_{preferred_family or 'pressure'}",
            "lifecycle": "frontier",
            "score": get_path(state, ["candidate", "scores", "active_frontier_accuracy"], None),
            "residual": None,
            "wall_type": "policy_selected_frontier",
            "frontier_source": f"{source}_synthetic",
            "pressure_card_id": pressure_card_id,
        }
    if not frontiers:
        return {}
    return max(frontiers, key=lambda row: float(row.get("residual") or 0.0))


def effective_frontier_selection(state: dict[str, Any]) -> tuple[str, str, str]:
    curriculum = state.get("benchmaxx_curriculum") if isinstance(state.get("benchmaxx_curriculum"), dict) else {}
    next_frontier = curriculum.get("next_frontier") if isinstance(curriculum.get("next_frontier"), dict) else {}
    runner = str(next_frontier.get("runner_family") or "")
    family = str(next_frontier.get("family") or "")
    runner_map = {
        "minecraft_rl_local": "minecraft_rl",
        "drone_rl_local": "drone_rl",
        "coding_local_sandbox": "coding_local_sandbox",
        "web_agent_local": "web_agent_local",
        "transfer_eval_local": "transfer_eval",
    }
    mapped = runner_map.get(runner, family)
    card = str(next_frontier.get("recommended_env") or "")
    if bool(next_frontier.get("runnable_now")) and mapped:
        return mapped, card, "benchmaxx_curriculum"
    policy = state.get("frontier_policy") if isinstance(state.get("frontier_policy"), dict) else {}
    return str(policy.get("frontier_family") or ""), str(policy.get("pressure_card_id") or ""), "frontier_policy_status"


def policy_matches_frontier(row: dict[str, Any], preferred_family: str, pressure_card_id: str) -> bool:
    name = str(row.get("benchmark_name") or "")
    row_family = frontier_family(row)
    if preferred_family and row_family != preferred_family:
        return False
    if not pressure_card_id:
        return True
    normalized_card = pressure_card_id.removeprefix("source_")
    return pressure_card_id in name or normalized_card in name


def frontier_family(row: dict[str, Any]) -> str:
    name = str(row.get("benchmark_name") or "")
    benchmark_type = str(row.get("benchmark_type") or "")
    if name.startswith("drone_rl_") or "drone_rl" in benchmark_type:
        return "drone_rl"
    if name.startswith("minecraft_rl_") or "minecraft" in name or "minecraft_rl" in benchmark_type:
        return "minecraft_rl"
    if name.startswith("coding_"):
        return "coding_local_sandbox"
    if name.startswith("web_agent_"):
        return "web_agent_local"
    if name.startswith("transfer_") or name.startswith("asi_transfer"):
        return "transfer_eval"
    if name.startswith("ocean-"):
        return "rl_local"
    if "babylm" in name:
        return "babylm_mutated"
    return "general"


def next_actions(plan: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("next_action")) for item in plan if item.get("status") == "ready"][:6]


def drone_sucker_for_frontier(frontier: dict[str, Any]) -> str:
    card = str(frontier.get("pressure_card_id") or frontier.get("benchmark_name") or "").lower()
    if "waypoint" in card or "pyflyt" in card:
        return "pyflyt_waypoint_sucker"
    if "grand_prix" in card or "sitl" in card or "mavsdk" in card:
        return "ai_grand_prix_sitl_sucker"
    return "gym_pybullet_hover_sucker"


def minecraft_sucker_for_frontier(frontier: dict[str, Any]) -> str:
    card = str(frontier.get("pressure_card_id") or frontier.get("benchmark_name") or "").lower()
    if "crafter" in card:
        return "crafter_bridge_sucker"
    if "java" in card or "local" in card or "minecraft" in card:
        return "minecraft_java_local_sucker"
    return "minecraft_open_world_sucker"


def latest_report(pattern: str) -> dict[str, Any]:
    reports = sorted((ROOT / "reports").glob(pattern), key=lambda item: item.stat().st_mtime)
    return read_json(reports[-1]) if reports else {}


def latest_drone_pressure() -> dict[str, Any]:
    reports = sorted(
        list((ROOT / "reports").glob("pressure_source_pyflyt_seed*.json"))
        + list((ROOT / "reports").glob("pressure_source_pyflyt_waypoints_seed*.json"))
        + list((ROOT / "reports").glob("pressure_source_gym_pybullet_drones_seed*.json")),
        key=lambda item: item.stat().st_mtime,
    )
    return read_json(reports[-1]) if reports else {}


def latest_minecraft_pressure() -> dict[str, Any]:
    reports = sorted(
        list((ROOT / "reports").glob("pressure_source_crafter_seed*.json"))
        + list((ROOT / "reports").glob("pressure_source_craftax_seed*.json"))
        + list((ROOT / "reports").glob("pressure_source_minedojo_seed*.json"))
        + list((ROOT / "reports").glob("pressure_source_malmo_seed*.json"))
        + list((ROOT / "reports").glob("pressure_source_minerl_seed*.json")),
        key=lambda item: item.stat().st_mtime,
    )
    return read_json(reports[-1]) if reports else {}


def latest_trace_manifest() -> dict[str, Any]:
    traces = sorted((ROOT / "reports" / "drone_traces").glob("*.manifest.json"), key=lambda item: item.stat().st_mtime)
    return read_json(traces[-1]) if traces else {}


def artifact_hints(state: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    for key in keys:
        value = state.get(key)
        if isinstance(value, dict) and value:
            hints[key] = {
                "score": value.get("score") or value.get("summary", {}).get("score"),
                "path": value.get("policy_path") or value.get("trace_path") or value.get("metrics", {}).get("trainer_report"),
                "created_utc": value.get("created_utc"),
            }
    return hints


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


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = [
        "# Arm Transfer Plan",
        "",
        f"Updated: {report.get('created_utc')}",
        "",
        f"Frontier family: {report.get('summary', {}).get('frontier_family')}",
        "",
        "## Transfer Edges",
        "",
    ]
    for item in report.get("transfer_plan", []):
        sucker = f" / sucker `{item.get('target_sucker')}`" if item.get("target_sucker") else ""
        rows.append(
            f"- {item.get('source_arm')} -> {item.get('target_arm')}{sucker}: {item.get('status')} - {item.get('hypothesis')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
