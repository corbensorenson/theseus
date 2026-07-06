"""Normalize legacy RL environment manifests into Theseus admission contracts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback.
    try:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CCA_REGISTRY = Path("D:/old_projects/cca/config/training/v1/rl_env_registry.json")
DEFAULT_CORBEN_REGISTRY = Path("D:/old_projects/corbens-trainer/registry/environments")
DEFAULT_DRONE_CAMPAIGN = Path("D:/old_projects/cca/config/training/v1/drone_sim2real_campaign_manifest.json")
DEFAULT_OUT = ROOT / "reports" / "legacy_rl_environment_admission.json"
DEFAULT_MARKDOWN_OUT = ROOT / "reports" / "legacy_rl_environment_admission.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cca-registry", default=str(DEFAULT_CCA_REGISTRY))
    parser.add_argument("--corben-registry", default=str(DEFAULT_CORBEN_REGISTRY))
    parser.add_argument("--drone-campaign", default=str(DEFAULT_DRONE_CAMPAIGN))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    if tomllib is None:
        raise SystemExit("tomllib/tomli is required to parse environment TOML manifests")

    cca_path = resolve_any(args.cca_registry)
    corben_path = resolve_any(args.corben_registry)
    drone_path = resolve_any(args.drone_campaign)
    cca_envs = load_cca_envs(cca_path)
    corben_envs = load_corben_envs(corben_path)
    drone_campaign = read_json(drone_path)
    environments = sorted(cca_envs + corben_envs, key=lambda row: (row["priority"], row["env_id"]))
    report = build_report(environments, cca_path, corben_path, drone_path, drone_campaign)

    write_json(resolve_any(args.out), report)
    write_text(resolve_any(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def load_cca_envs(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("environments") if isinstance(payload.get("environments"), list) else []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        env_id = str(row.get("id") or "")
        local_path = str(row.get("local_path") or "")
        out.append(
            normalize_env(
                env_id=env_id,
                source_project="cca",
                source_path=path,
                family=str(row.get("family") or infer_family(env_id)),
                adapter=str(row.get("adapter") or infer_adapter(env_id)),
                scenario=str(row.get("scenario") or env_id),
                lane=infer_lane(env_id),
                install_burden=str(row.get("install_burden") or "unknown"),
                local_path=local_path,
                trainable=bool(row.get("trainable", True)),
                evaluation_only=bool(row.get("eval_only", False)),
                deterministic_replay=True,
                action_space=infer_action_space(env_id),
                observation_space=infer_observation_space(env_id),
                benchmark_eligibility="legacy_recipe",
                tags=listify(row.get("tags")),
            )
        )
    return out


def load_corben_envs(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for manifest in sorted(path.glob("*.toml")):
        try:
            raw = tomllib.loads(manifest.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - bad legacy manifests are skipped but not fatal.
            continue
        common = raw.get("common") if isinstance(raw.get("common"), dict) else {}
        metadata = common.get("metadata") if isinstance(common.get("metadata"), dict) else {}
        install = raw.get("install") if isinstance(raw.get("install"), dict) else {}
        capabilities = raw.get("capabilities") if isinstance(raw.get("capabilities"), dict) else {}
        execution = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}
        env_id = str(common.get("id") or manifest.stem)
        out.append(
            normalize_env(
                env_id=env_id,
                source_project="corbens-trainer",
                source_path=manifest,
                family=str(raw.get("family") or infer_family(env_id)),
                adapter=str(metadata.get("adapter") or infer_adapter(env_id)),
                scenario=str(metadata.get("scenario") or env_id),
                lane=str(execution.get("lane") or infer_lane(env_id)),
                install_burden=str(install.get("burden") or "unknown"),
                local_path="",
                trainable=not bool(raw.get("evaluation_only", False)),
                evaluation_only=bool(raw.get("evaluation_only", False)),
                deterministic_replay=bool(raw.get("deterministic_seeds", False))
                or bool(capabilities.get("deterministic_replay", False)),
                action_space=str(capabilities.get("action_space") or infer_action_space(env_id)),
                observation_space=str(capabilities.get("observation_space") or infer_observation_space(env_id)),
                benchmark_eligibility=str(raw.get("benchmark_eligibility") or "legacy_recipe"),
                tags=listify(common.get("tags")),
                local_laptop_allowed=execution.get("local_laptop_allowed"),
                min_cpu_cores=execution.get("min_cpu_cores"),
                min_ram_gb=execution.get("min_ram_gb"),
                preferred_parallel_envs=execution.get("preferred_parallel_envs"),
            )
        )
    return out


def normalize_env(
    *,
    env_id: str,
    source_project: str,
    source_path: Path,
    family: str,
    adapter: str,
    scenario: str,
    lane: str,
    install_burden: str,
    local_path: str,
    trainable: bool,
    evaluation_only: bool,
    deterministic_replay: bool,
    action_space: str,
    observation_space: str,
    benchmark_eligibility: str,
    tags: list[str],
    local_laptop_allowed: Any = None,
    min_cpu_cores: Any = None,
    min_ram_gb: Any = None,
    preferred_parallel_envs: Any = None,
) -> dict[str, Any]:
    priority = infer_priority(env_id, install_burden)
    local_exists = path_exists(local_path)
    admission_state = infer_admission_state(env_id, install_burden, local_exists, source_project)
    safety_gates = safety_gates_for(env_id)
    return {
        "env_id": env_id,
        "source_project": source_project,
        "source_path": rel_or_abs(source_path),
        "family": family,
        "lane": lane,
        "adapter": adapter,
        "scenario": scenario,
        "priority": priority,
        "admission_state": admission_state,
        "trainable": trainable,
        "evaluation_only": evaluation_only,
        "benchmark_eligibility": benchmark_eligibility,
        "install_burden": install_burden,
        "dependency_state": "local_dependency_present" if local_exists else "manifest_ready_dependency_missing",
        "local_path": local_path,
        "action_schema": {"space": action_space, "contract": action_contract(env_id, action_space)},
        "observation_schema": {"space": observation_space, "contract": observation_contract(env_id, observation_space)},
        "reward_schema": reward_contract(env_id),
        "deterministic_replay_required": True,
        "deterministic_replay_declared": deterministic_replay,
        "smoke_command_required": True,
        "local_laptop_allowed": local_laptop_allowed,
        "min_cpu_cores": min_cpu_cores,
        "min_ram_gb": min_ram_gb,
        "preferred_parallel_envs": preferred_parallel_envs,
        "tags": tags,
        "safety_gates": safety_gates,
        "external_inference_calls": 0,
    }


def build_report(
    environments: list[dict[str, Any]],
    cca_path: Path,
    corben_path: Path,
    drone_path: Path,
    drone_campaign: dict[str, Any],
) -> dict[str, Any]:
    by_state = Counter(str(row.get("admission_state")) for row in environments)
    by_source = Counter(str(row.get("source_project")) for row in environments)
    p0 = [row for row in environments if row.get("priority") == "P0"]
    drone_envs = [row for row in environments if "drone" in str(row.get("env_id"))]
    hardware_envs = [
        row
        for row in drone_envs
        if ".practice." in str(row.get("env_id")) or ".competition." in str(row.get("env_id"))
    ]
    gates = [
        gate("cca_env_registry_present", cca_path.exists(), rel_or_abs(cca_path)),
        gate("corbens_trainer_env_manifests_present", corben_path.exists(), rel_or_abs(corben_path)),
        gate("normalized_environment_count_nonzero", len(environments) > 0, f"environments={len(environments)}"),
        gate("p0_smoke_lane_defined", len(p0) > 0, [row["env_id"] for row in p0[:12]]),
        gate("drone_campaign_contract_present", bool(drone_campaign), rel_or_abs(drone_path)),
        gate(
            "hardware_envs_blocked_from_autopromotion",
            all(row.get("admission_state") == "hardware_gated" for row in hardware_envs),
            [row["env_id"] for row in hardware_envs],
        ),
        gate("external_inference_zero", True, "local manifest normalization only"),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    return {
        "policy": "project_theseus_legacy_rl_environment_admission_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "sources": {
            "cca_registry": rel_or_abs(cca_path),
            "corbens_trainer_registry": rel_or_abs(corben_path),
            "drone_campaign": rel_or_abs(drone_path),
        },
        "summary": {
            "environments": len(environments),
            "by_source": dict(by_source),
            "by_admission_state": dict(by_state),
            "p0_smoke_lane": len(p0),
            "drone_envs": len(drone_envs),
            "hardware_gated_envs": len(hardware_envs),
            "external_inference_calls": 0,
        },
        "recommended_smoke_order": [compact_env(row) for row in p0[:16]],
        "drone_safety_contract": {
            "campaign_id": drone_campaign.get("campaign_id"),
            "bringup_ladder": drone_campaign.get("bringup_ladder") or [],
            "safety_and_parity_gates": drone_campaign.get("safety_and_parity_gates") or {},
            "blackbox_requirements": drone_campaign.get("blackbox_requirements") or {},
            "hard_rule": "practice and competition drone lanes remain hardware_gated until sim smoke, parity, and bounded-flight evidence exist",
        },
        "environments": environments,
        "gates": gates,
        "external_inference_calls": 0,
    }


def infer_admission_state(env_id: str, burden: str, local_exists: bool, source_project: str) -> str:
    text = env_id.lower()
    if ".practice." in text or ".competition." in text or "tello" in text or "px4" in text:
        return "hardware_gated"
    if "drone.sim" in text or "flightmare" in text or "pybullet_drones" in text or "airgym" in text:
        return "sim_recipe_pending_dependency" if not local_exists else "sim_smoke_candidate"
    if local_exists:
        return "smoke_candidate"
    if burden in {"lightweight", "moderate"} or source_project == "corbens-trainer":
        return "manifest_ready_pending_dependency"
    return "recipe_pending_dependency"


def infer_priority(env_id: str, burden: str) -> str:
    text = env_id.lower()
    if any(token in text for token in ["cartpole", "minigrid", "textworld", "scienceworld", "crafter", "procgen"]):
        return "P0"
    if any(token in text for token in ["browsergym", "webarena", "osworld", "appworld", "swebench", "bigcode", "humaneval"]):
        return "P1"
    if any(token in text for token in ["drone.sim", "dm_control", "metaworld", "alfworld"]):
        return "P1"
    if any(token in text for token in ["drone.practice", "drone.competition", "tello", "px4"]):
        return "P2"
    if burden == "lightweight":
        return "P1"
    return "P2"


def infer_family(env_id: str) -> str:
    text = env_id.lower()
    for token, family in [
        ("browser", "agentic_web"),
        ("webarena", "agentic_web"),
        ("osworld", "desktop_agent"),
        ("android", "mobile_agent"),
        ("drone", "aerial_control"),
        ("swe", "code_repair"),
        ("bigcode", "code_repair"),
        ("humaneval", "code_generation"),
        ("math", "reasoning"),
        ("gsm8k", "reasoning"),
        ("textworld", "text_game"),
        ("alfworld", "embodied_text"),
        ("scienceworld", "science_text"),
        ("minigrid", "gridworld"),
        ("procgen", "procedural_control"),
        ("crafter", "survival_crafting"),
        ("board", "self_play"),
        ("open_spiel", "self_play"),
    ]:
        if token in text:
            return family
    return "legacy_rl"


def infer_lane(env_id: str) -> str:
    text = env_id.lower()
    if any(token in text for token in ["browser", "webarena", "osworld", "android", "webgym", "miniwob"]):
        return "agentic_web_desktop"
    if any(token in text for token in ["swe", "bigcode", "humaneval"]):
        return "code_repair_rl"
    if "drone" in text or "tello" in text or "px4" in text:
        return "aerial_sim2real"
    if any(token in text for token in ["textworld", "alfworld", "scienceworld"]):
        return "language_grounded_env"
    return "rl_control"


def infer_adapter(env_id: str) -> str:
    text = env_id.lower()
    if "browsergym" in text:
        return "browsergym-compat"
    if "webarena" in text:
        return "webarena-compat"
    if "osworld" in text:
        return "osworld-compat"
    if "swe" in text:
        return "swebench-compat"
    if "drone" in text or "tello" in text or "px4" in text:
        return "aerial-bridge"
    if any(token in text for token in ["cartpole", "minigrid", "procgen", "dm_control", "metaworld"]):
        return "gymnasium-compat"
    return "legacy-env-adapter"


def infer_action_space(env_id: str) -> str:
    text = env_id.lower()
    if any(token in text for token in ["drone", "tello", "px4", "dm_control", "metaworld"]):
        return "continuous_control"
    if any(token in text for token in ["browser", "webarena", "osworld", "android", "webgym"]):
        return "tool_ui_actions"
    if any(token in text for token in ["swe", "bigcode", "humaneval"]):
        return "patch_or_code_actions"
    return "discrete"


def infer_observation_space(env_id: str) -> str:
    text = env_id.lower()
    if any(token in text for token in ["browser", "webarena", "osworld", "android", "webgym"]):
        return "ui_dom_screenshot_text"
    if any(token in text for token in ["drone", "tello", "px4"]):
        return "telemetry_video_command_ack"
    if any(token in text for token in ["swe", "bigcode", "humaneval"]):
        return "repo_issue_tests_trace"
    if any(token in text for token in ["textworld", "alfworld", "scienceworld"]):
        return "text_state"
    return "state"


def action_contract(env_id: str, action_space: str) -> str:
    if "drone" in env_id.lower() or "tello" in env_id.lower() or "px4" in env_id.lower():
        return "bounded residual commands over stable controller; command and ack streams logged"
    if "patch" in action_space or "code" in action_space:
        return "patch proposal, test command, and replay row"
    if "ui" in action_space:
        return "typed UI action with target, screenshot/DOM reference, and replay id"
    return "seeded environment action with deterministic replay row"


def observation_contract(env_id: str, observation_space: str) -> str:
    if "drone" in env_id.lower() or "telemetry" in observation_space:
        return "telemetry, video hash, command ack, bridge profile hash, environment contract hash"
    if "repo" in observation_space:
        return "issue text, file observations, test receipts, and hidden-answer exclusion state"
    if "ui" in observation_space:
        return "screen/DOM/text observation with privacy and network policy markers"
    return "seeded state observation with episode id and step index"


def reward_contract(env_id: str) -> dict[str, Any]:
    text = env_id.lower()
    if "drone" in text:
        return {
            "primary": "safety_bounded_task_progress",
            "must_log": ["terminal", "collision_or_safety_event", "blackbox_parity"],
        }
    if any(token in text for token in ["swe", "bigcode", "humaneval"]):
        return {"primary": "test_grounded_correctness", "must_log": ["tests_run", "patch_hash", "public_claim_forbidden"]}
    if any(token in text for token in ["browser", "webarena", "osworld", "android"]):
        return {"primary": "task_completion_with_replay", "must_log": ["action_trace", "final_state", "side_effects"]}
    return {"primary": "environment_score", "must_log": ["episode_id", "step", "terminal", "return"]}


def safety_gates_for(env_id: str) -> list[str]:
    text = env_id.lower()
    gates = ["deterministic_replay_id", "observation_action_schema", "reward_schema", "external_inference_zero"]
    if "drone" in text or "tello" in text or "px4" in text:
        gates.extend(["sim_smoke_before_hardware", "blackbox_parity", "bounded_control", "human_review_before_device"])
    if any(token in text for token in ["swe", "bigcode", "humaneval", "benchmark"]):
        gates.extend(["holdout_boundary", "no_reference_answer_training", "private_claim_only_until_fresh_run"])
    return gates


def compact_env(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "env_id": row.get("env_id"),
        "source_project": row.get("source_project"),
        "priority": row.get("priority"),
        "admission_state": row.get("admission_state"),
        "adapter": row.get("adapter"),
        "family": row.get("family"),
        "next_step": "install dependency or map to existing local adapter, then run seeded smoke with replay",
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def path_exists(local_path: str) -> bool:
    if not local_path:
        return False
    path = Path(local_path)
    candidates = [path] if path.is_absolute() else [ROOT / path, Path("D:/old_projects/cca") / path]
    return any(candidate.exists() for candidate in candidates)


def listify(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value:
        return [str(value)]
    return []


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Legacy RL Environment Admission",
        "",
        f"- Trigger state: `{report['trigger_state']}`",
        f"- Environments: `{summary['environments']}`",
        f"- P0 smoke lane candidates: `{summary['p0_smoke_lane']}`",
        f"- Drone envs: `{summary['drone_envs']}`",
        f"- Hardware gated envs: `{summary['hardware_gated_envs']}`",
        "",
        "## Recommended Smoke Order",
    ]
    for row in report.get("recommended_smoke_order", []):
        lines.append(f"- `{row['env_id']}` state=`{row['admission_state']}` adapter=`{row['adapter']}`")
    lines.extend(["", "## Gates"])
    for row in report.get("gates", []):
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- `{mark}` `{row['gate']}`: {row['evidence']}")
    lines.append("")
    return "\n".join(lines)


def resolve_any(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
