"""Build executable smoke contracts for admitted legacy RL environments."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADMISSION = ROOT / "reports" / "legacy_rl_environment_admission.json"
DEFAULT_OUT = ROOT / "reports" / "legacy_rl_smoke_plan.json"
DEFAULT_PLAN_OUT = ROOT / "data" / "rl_smoke" / "legacy_rl_smoke_plan.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", default=str(DEFAULT_ADMISSION.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--plan-out", default=str(DEFAULT_PLAN_OUT.relative_to(ROOT)))
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()

    admission_path = resolve(args.admission)
    admission = read_json(admission_path)
    report = build_report(
        admission=admission,
        admission_path=admission_path,
        plan_out=resolve(args.plan_out),
        limit=max(1, args.limit),
    )
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(*, admission: dict[str, Any], admission_path: Path, plan_out: Path, limit: int) -> dict[str, Any]:
    envs = [row for row in admission.get("environments", []) if isinstance(row, dict)]
    selected = select_envs(envs, limit)
    plans = [plan_for_env(row, index + 1) for index, row in enumerate(selected)]
    write_jsonl(plan_out, plans)
    by_state = Counter(str(row.get("smoke_state")) for row in plans)
    blocked_hardware = [
        row for row in plans if row.get("hardware_gated") and row.get("smoke_state") == "hardware_gated_not_executable"
    ]
    executable = [row for row in plans if row.get("smoke_state") == "ready_for_seeded_smoke"]
    fallback_executable = [row for row in executable if row.get("runner_mode") == "vendored_contract_fallback"]
    gates = [
        gate("admission_report_present", bool(admission), rel_or_abs(admission_path)),
        gate("plan_rows_written", bool(plans), f"rows={len(plans)} path={rel(plan_out)}"),
        gate("p0_envs_considered", any(row.get("priority") == "P0" for row in plans), [row.get("env_id") for row in plans[:12]]),
        gate("hardware_envs_not_executable", len(blocked_hardware) == len([row for row in plans if row.get("hardware_gated")]), [row.get("env_id") for row in blocked_hardware]),
        gate("all_rows_have_observation_action_reward_contracts", all(has_contracts(row) for row in plans), "schema contracts present"),
        gate("external_inference_zero", True, "local dependency probes only"),
    ]
    trigger_state = "GREEN" if executable and all(row["passed"] for row in gates) else "YELLOW"
    if not plans or not all(row["passed"] for row in gates):
        trigger_state = "RED"
    return {
        "policy": "project_theseus_legacy_rl_smoke_plan_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "admission_report": rel_or_abs(admission_path),
        "plan_path": rel(plan_out),
        "summary": {
            "planned_envs": len(plans),
            "ready_for_seeded_smoke": len(executable),
            "vendored_contract_fallback_ready": len(fallback_executable),
            "pending_dependency": by_state.get("pending_dependency", 0),
            "source_present_pending_install": by_state.get("source_present_pending_install", 0),
            "runner_pending_adapter": by_state.get("runner_pending_adapter", 0),
            "hardware_gated_not_executable": by_state.get("hardware_gated_not_executable", 0),
            "by_smoke_state": dict(by_state),
            "external_inference_calls": 0,
        },
        "plans": plans,
        "gates": gates,
        "next_actions": next_actions(plans),
        "external_inference_calls": 0,
    }


def select_envs(envs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    priority_rank = {"P0": 0, "P1": 1, "P2": 2}
    state_rank = {
        "smoke_candidate": 0,
        "sim_smoke_candidate": 1,
        "manifest_ready_pending_dependency": 2,
        "recipe_pending_dependency": 3,
        "hardware_gated": 4,
    }
    return sorted(
        envs,
        key=lambda row: (
            priority_rank.get(str(row.get("priority") or "P2"), 9),
            state_rank.get(str(row.get("admission_state") or ""), 9),
            str(row.get("env_id") or ""),
        ),
    )[:limit]


def plan_for_env(env: dict[str, Any], ordinal: int) -> dict[str, Any]:
    env_id = str(env.get("env_id") or f"env_{ordinal}")
    modules = modules_for(env)
    probes = [
        {
            "module": module,
            "available": bool(importlib.util.find_spec(module)),
        }
        for module in modules
    ]
    hardware_gated = str(env.get("admission_state")) == "hardware_gated"
    module_ready = all(item["available"] for item in probes) if probes else False
    local_path = str(env.get("local_path") or "")
    local_path_present = local_path_exists(local_path)
    runner_supported = runner_supports(env)
    fallback_supported = vendored_contract_fallback_supported(env)
    runner_mode = "native_adapter"
    if hardware_gated:
        smoke_state = "hardware_gated_not_executable"
    elif module_ready and runner_supported:
        smoke_state = "ready_for_seeded_smoke"
    elif fallback_supported:
        smoke_state = "ready_for_seeded_smoke"
        runner_mode = "vendored_contract_fallback"
    elif module_ready:
        smoke_state = "runner_pending_adapter"
    elif local_path_present:
        smoke_state = "source_present_pending_install"
    else:
        smoke_state = "pending_dependency"
    output_report = f"reports/legacy_rl_smokes/{safe_name(env_id)}_seed{17 + ordinal}.json"
    commands = commands_for(
        env,
        smoke_state=smoke_state,
        runner_mode=runner_mode,
        seed=17 + ordinal,
        step_budget=16,
        output_report=output_report,
    )
    required_evidence = [
        "episode_id",
        "seed",
        "runner_mode",
        "reset_receipt",
        "step_receipts",
        "terminal_or_step_budget",
        "return_value",
        "observation_hashes",
        "action_hashes",
        "external_inference_zero",
    ]
    if runner_mode == "vendored_contract_fallback":
        required_evidence.extend(
            [
                "vendored_contract_fallback_disclosed",
                "benchmark_score_claim_allowed_false",
                "contract_hash",
            ]
        )
    return {
        "plan_id": f"legacy_rl_smoke_{ordinal:03d}_{safe_name(env_id)}",
        "env_id": env_id,
        "priority": env.get("priority"),
        "source_project": env.get("source_project"),
        "family": env.get("family"),
        "adapter": env.get("adapter"),
        "scenario": env.get("scenario"),
        "admission_state": env.get("admission_state"),
        "smoke_state": smoke_state,
        "runner_mode": runner_mode,
        "hardware_gated": hardware_gated,
        "dependency_probes": probes,
        "vendored_contract_fallback_available": fallback_supported,
        "local_path": local_path,
        "local_path_present": local_path_present,
        "seed": 17 + ordinal,
        "episode_budget": 1,
        "step_budget": 16,
        "commands": commands,
        "required_evidence": required_evidence,
        "action_schema": env.get("action_schema") or {},
        "observation_schema": env.get("observation_schema") or {},
        "reward_schema": env.get("reward_schema") or {},
        "safety_gates": env.get("safety_gates") or [],
        "output_report": output_report,
        "external_inference_calls": 0,
    }


def modules_for(env: dict[str, Any]) -> list[str]:
    env_id = str(env.get("env_id") or "").lower()
    adapter = str(env.get("adapter") or "").lower()
    scenario = str(env.get("scenario") or "").lower()
    text = " ".join([env_id, adapter, scenario])
    pairs = [
        ("minigrid", "minigrid"),
        ("cartpole", "gymnasium"),
        ("gymnasium", "gymnasium"),
        ("procgen", "procgen"),
        ("crafter", "crafter"),
        ("textworld", "textworld"),
        ("scienceworld", "scienceworld"),
        ("alfworld", "alfworld"),
        ("browsergym", "browsergym"),
        ("webarena", "browsergym"),
        ("osworld", "osworld"),
        ("dm_control", "dm_control"),
        ("metaworld", "metaworld"),
        ("pettingzoo", "pettingzoo"),
        ("appworld", "appworld"),
    ]
    modules = [module for token, module in pairs if token in text]
    return list(dict.fromkeys(modules))


def runner_supports(env: dict[str, Any]) -> bool:
    adapter = str(env.get("adapter") or "").lower()
    scenario = str(env.get("scenario") or "")
    if "gymnasium" in adapter and scenario and not scenario.startswith("rl_env."):
        return True
    return False


def vendored_contract_fallback_supported(env: dict[str, Any]) -> bool:
    if str(env.get("admission_state")) == "hardware_gated":
        return False
    return has_contracts(env)


def commands_for(
    env: dict[str, Any],
    *,
    smoke_state: str,
    runner_mode: str,
    seed: int,
    step_budget: int,
    output_report: str,
) -> list[dict[str, Any]]:
    env_id = str(env.get("env_id") or "")
    adapter = str(env.get("adapter") or "")
    command = [
        "python",
        "scripts/legacy_rl_seeded_smoke.py",
        "--env-id",
        env_id,
        "--adapter",
        adapter,
        "--scenario",
        str(env.get("scenario") or env_id),
        "--seed",
        str(seed),
        "--step-budget",
        str(step_budget),
        "--out",
        output_report,
    ]
    if runner_mode == "vendored_contract_fallback":
        command.append("--allow-contract-fallback")
    execute_now = smoke_state == "ready_for_seeded_smoke"
    why_not_now = {
        "hardware_gated_not_executable": "hardware-gated environments require sim parity and bounded-flight evidence first",
        "source_present_pending_install": "local source exists, but adapter dependencies are not importable yet",
        "pending_dependency": "adapter dependencies are not importable yet",
        "runner_pending_adapter": "dependencies are importable, but this adapter still needs a concrete seeded-smoke runner",
    }.get(smoke_state)
    row = {
        "kind": "seeded_smoke" if execute_now else "future_seeded_smoke",
        "command": command,
        "execute_now": execute_now,
        "runner_mode": runner_mode,
    }
    if execute_now and runner_mode == "vendored_contract_fallback":
        row["fallback_disclosure"] = "executes deterministic local contract shim; does not claim real environment benchmark score"
    if why_not_now:
        row["why_not_now"] = why_not_now
    return [
        row
    ]


def next_actions(plans: list[dict[str, Any]]) -> list[str]:
    missing = []
    for plan in plans:
        if plan.get("runner_mode") == "vendored_contract_fallback":
            continue
        if plan.get("smoke_state") == "runner_pending_adapter":
            missing.append(f"{plan.get('env_id')}: implement/map seeded-smoke runner for {plan.get('adapter')}")
            continue
        if plan.get("smoke_state") not in {"pending_dependency", "source_present_pending_install"}:
            continue
        modules = [probe["module"] for probe in plan.get("dependency_probes", []) if not probe.get("available")]
        if modules:
            missing.append(f"{plan.get('env_id')}: install/provide {', '.join(modules)}")
    actions = missing[:8]
    if any(plan.get("hardware_gated") for plan in plans):
        actions.append("Keep Tello/PX4 hardware lanes gated until sim smoke, blackbox parity, and bounded-flight evidence exist.")
    if not actions:
        actions.append("Run the ready seeded smoke commands and persist receipts under reports/legacy_rl_smokes/.")
    return actions


def has_contracts(row: dict[str, Any]) -> bool:
    return bool(row.get("action_schema") and row.get("observation_schema") and row.get("reward_schema"))


def local_path_exists(local_path: str) -> bool:
    if not local_path:
        return False
    path = Path(local_path)
    candidates = [path] if path.is_absolute() else [ROOT / path, Path("D:/old_projects/cca") / path]
    return any(candidate.exists() for candidate in candidates)


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in value)[:120]


def resolve(value: str) -> Path:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
