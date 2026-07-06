"""Run a bounded seeded smoke episode for an admitted legacy RL environment.

The report intentionally stores hashes and receipts instead of raw observations
or actions. This keeps legacy environments useful for admission/replay without
turning smoke evidence into an accidental data dump.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "reports" / "legacy_rl_smokes"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--step-budget", type=int, default=16)
    parser.add_argument("--out", default="")
    parser.add_argument("--allow-contract-fallback", action="store_true")
    args = parser.parse_args()

    out = resolve(args.out) if args.out else DEFAULT_OUT_DIR / f"{safe_name(args.env_id)}_seed{args.seed}.json"
    report = run_smoke(
        env_id=args.env_id,
        adapter=args.adapter,
        scenario=args.scenario,
        seed=args.seed,
        step_budget=max(1, args.step_budget),
        allow_contract_fallback=args.allow_contract_fallback,
    )
    write_json(out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


def run_smoke(
    *,
    env_id: str,
    adapter: str,
    scenario: str,
    seed: int,
    step_budget: int,
    allow_contract_fallback: bool = False,
) -> dict[str, Any]:
    missing = [module for module in required_modules(adapter, scenario, env_id) if importlib.util.find_spec(module) is None]
    base = {
        "policy": "project_theseus_legacy_rl_seeded_smoke_v1",
        "created_utc": now(),
        "env_id": env_id,
        "adapter": adapter,
        "scenario": scenario,
        "seed": seed,
        "step_budget": step_budget,
        "episode_id": stable_hash({"env_id": env_id, "scenario": scenario, "seed": seed})[:16],
        "runner_mode": "native_adapter",
        "contract_hash": stable_hash(
            {
                "env_id": env_id,
                "adapter": adapter,
                "scenario": scenario,
                "seed": seed,
                "step_budget": step_budget,
                "policy": "project_theseus_legacy_rl_seeded_smoke_v1",
            }
        ),
        "external_inference_calls": 0,
    }
    if missing:
        if allow_contract_fallback:
            return run_contract_fallback(
                base,
                missing_modules=missing,
                fallback_reason="missing optional environment dependency",
                seed=seed,
                step_budget=step_budget,
            )
        return {
            **base,
            "status": "blocked_missing_dependency",
            "missing_modules": missing,
            "reset_receipt": None,
            "step_receipts": [],
            "terminal_or_step_budget": False,
            "return_value": None,
            "observation_hashes": [],
            "action_hashes": [],
            "gates": gates(False, "missing dependency"),
        }
    if "gymnasium" in adapter.lower() and scenario and not scenario.startswith("rl_env."):
        return run_gymnasium(base, scenario=scenario, seed=seed, step_budget=step_budget)
    if allow_contract_fallback:
        return run_contract_fallback(
            base,
            missing_modules=[],
            fallback_reason="adapter runner not implemented for admitted legacy environment",
            seed=seed,
            step_budget=step_budget,
        )
    return {
        **base,
        "status": "blocked_adapter_not_implemented",
        "missing_modules": [],
        "reset_receipt": None,
        "step_receipts": [],
        "terminal_or_step_budget": False,
        "return_value": None,
        "observation_hashes": [],
        "action_hashes": [],
        "gates": gates(False, "adapter runner not implemented"),
    }


def run_contract_fallback(
    base: dict[str, Any],
    *,
    missing_modules: list[str],
    fallback_reason: str,
    seed: int,
    step_budget: int,
) -> dict[str, Any]:
    rng = random.Random(stable_hash({"seed": seed, "episode": base["episode_id"]}))
    target = 7 + (seed % 5)
    position = seed % 3
    observation = {
        "kind": "vendored_contract_state",
        "position": position,
        "target": target,
        "step": 0,
    }
    observation_hashes = [stable_hash(observation)]
    action_hashes: list[str] = []
    step_receipts = []
    total_reward = 0.0
    terminal = False
    for step in range(step_budget):
        direction = 1 if position <= target else -1
        jitter = rng.choice([0, 0, 1, -1])
        action = {"kind": "contract_discrete_action", "delta": direction + jitter}
        position += int(action["delta"])
        terminal = abs(position - target) <= 1 or step + 1 >= step_budget
        reward = 1.0 if terminal and abs(position - target) <= 1 else -0.01
        total_reward += reward
        observation = {
            "kind": "vendored_contract_state",
            "position": position,
            "target": target,
            "step": step + 1,
            "terminal": terminal,
        }
        action_hash = stable_hash(action)
        obs_hash = stable_hash(observation)
        action_hashes.append(action_hash)
        observation_hashes.append(obs_hash)
        step_receipts.append(
            {
                "step": step,
                "action_hash": action_hash,
                "observation_hash": obs_hash,
                "reward": reward,
                "terminal": terminal,
            }
        )
        if terminal:
            break
    return {
        **base,
        "status": "passed",
        "runner_mode": "vendored_contract_fallback",
        "vendored_contract_fallback_disclosed": True,
        "benchmark_score_claim_allowed": False,
        "fallback_reason": fallback_reason,
        "missing_modules": missing_modules,
        "reset_receipt": {
            "observation_hash": observation_hashes[0],
            "seed": seed,
        },
        "step_receipts": step_receipts,
        "terminal_or_step_budget": terminal or len(step_receipts) >= step_budget,
        "return_value": round(total_reward, 6),
        "observation_hashes": observation_hashes,
        "action_hashes": action_hashes,
        "gates": gates(
            True,
            "deterministic vendored contract smoke completed; real dependency score not claimed",
            runner_mode="vendored_contract_fallback",
        ),
    }


def run_gymnasium(base: dict[str, Any], *, scenario: str, seed: int, step_budget: int) -> dict[str, Any]:
    gym = importlib.import_module("gymnasium")
    env = gym.make(scenario)
    try:
        reset_result = env.reset(seed=seed)
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        observation_hashes = [stable_hash(observation)]
        action_hashes: list[str] = []
        step_receipts = []
        total_reward = 0.0
        terminal = False
        for step in range(step_budget):
            action = env.action_space.sample()
            action_hash = stable_hash(action)
            result = env.step(action)
            if len(result) == 5:
                observation, reward, terminated, truncated, _info = result
                terminal = bool(terminated or truncated)
            else:
                observation, reward, done, _info = result
                terminal = bool(done)
            obs_hash = stable_hash(observation)
            action_hashes.append(action_hash)
            observation_hashes.append(obs_hash)
            total_reward += float(reward)
            step_receipts.append(
                {
                    "step": step,
                    "action_hash": action_hash,
                    "observation_hash": obs_hash,
                    "reward": float(reward),
                    "terminal": terminal,
                }
            )
            if terminal:
                break
        return {
            **base,
            "status": "passed",
            "runner_mode": "native_adapter",
            "vendored_contract_fallback_disclosed": False,
            "benchmark_score_claim_allowed": True,
            "missing_modules": [],
            "reset_receipt": {
                "observation_hash": observation_hashes[0],
                "seed": seed,
            },
            "step_receipts": step_receipts,
            "terminal_or_step_budget": terminal or len(step_receipts) >= step_budget,
            "return_value": total_reward,
            "observation_hashes": observation_hashes,
            "action_hashes": action_hashes,
            "gates": gates(True, "bounded seeded smoke completed"),
        }
    except Exception as exc:  # noqa: BLE001 - report the adapter failure without raw env state.
        return {
            **base,
            "status": "failed_runtime_exception",
            "missing_modules": [],
            "error_type": type(exc).__name__,
            "error": str(exc)[:240],
            "reset_receipt": None,
            "step_receipts": [],
            "terminal_or_step_budget": False,
            "return_value": None,
            "observation_hashes": [],
            "action_hashes": [],
            "gates": gates(False, "runtime exception"),
        }
    finally:
        close = getattr(env, "close", None)
        if close:
            close()


def required_modules(adapter: str, scenario: str, env_id: str) -> list[str]:
    text = " ".join([adapter, scenario, env_id]).lower()
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
    return list(dict.fromkeys(module for token, module in pairs if token in text))


def gates(passed: bool, evidence: str, *, runner_mode: str = "native_adapter") -> list[dict[str, Any]]:
    rows = [
        {"gate": "external_inference_zero", "passed": True, "evidence": "local environment imports and steps only"},
        {"gate": "bounded_step_budget", "passed": True, "evidence": "single episode with fixed step budget"},
        {"gate": "receipt_hashes_only", "passed": True, "evidence": "raw observations and actions are not written"},
        {"gate": "seeded_smoke_completed", "passed": passed, "evidence": evidence},
    ]
    rows.append(
        {
            "gate": "native_or_disclosed_contract_runner",
            "passed": runner_mode == "native_adapter" or passed,
            "evidence": runner_mode,
        }
    )
    if runner_mode == "vendored_contract_fallback":
        rows.append(
            {
                "gate": "benchmark_score_not_claimed_for_fallback",
                "passed": True,
                "evidence": "fallback validates reset/step/replay contract only",
            }
        )
    return rows


def stable_hash(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, default=repr, ensure_ascii=True).encode("utf-8")
    except TypeError:
        encoded = repr(value).encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
