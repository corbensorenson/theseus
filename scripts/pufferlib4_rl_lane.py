"""Governed PufferLib 4 RL lane for Theseus.

The lane prefers fast Puffer/Ocean rollouts when the native backend is present.
Until then it still contributes transferable RL control capsules and records a
specific residual, so the work board learns from the blocker instead of idling
or accidentally downloading ROMs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import theseus_runtime


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUNTIME_PATHS = theseus_runtime.runtime_report(create=False)["paths"]
DEFAULT_RUNTIME_ROOT = Path(RUNTIME_PATHS["runtime_root"]["path"])
DEFAULT_DATA_DIR = Path(RUNTIME_PATHS["data_dir"]["path"])
DEFAULT_CACHE_DIR = Path(RUNTIME_PATHS["cache_dir"]["path"])
D_DATA = DEFAULT_DATA_DIR / "rl" / "pufferlib4"
DEFAULT_OUT = REPORTS / "pufferlib4_rl_lane.json"
DEFAULT_MARKDOWN = REPORTS / "pufferlib4_rl_lane.md"
DEFAULT_CAPSULES = D_DATA / "pufferlib4_rl_sts_capsules.jsonl"
DEFAULT_POLICY_TRACE = D_DATA / "pufferlib4_native_policy_trace.jsonl"
DEFAULT_PROBE = REPORTS / "pufferlib4_capability_probe.json"
DEFAULT_PROBE_MD = REPORTS / "pufferlib4_capability_probe.md"
PUFFER_VENV_PY = ROOT / ".venv-puffer" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENDOR_PUFFER = ROOT / "vendor" / "pufferlib"
D_TMP = DEFAULT_RUNTIME_ROOT / "tmp" / "pufferlib4"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--capsules-out", default=str(DEFAULT_CAPSULES))
    parser.add_argument("--probe-out", default=str(DEFAULT_PROBE.relative_to(ROOT)))
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--policy-trace-out", default=str(DEFAULT_POLICY_TRACE))
    parser.add_argument("--skip-policy-smoke", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    REPORTS.mkdir(parents=True, exist_ok=True)
    D_DATA.mkdir(parents=True, exist_ok=True)

    probe_path = resolve(args.probe_out)
    if args.probe or not probe_path.exists():
        run_probe(probe_path)
    probe = read_json(probe_path)
    lane = build_lane_report(
        probe,
        run_smoke=args.run_smoke,
        skip_policy_smoke=args.skip_policy_smoke,
        policy_trace_out=resolve(args.policy_trace_out),
    )
    lane["elapsed_seconds"] = round(time.perf_counter() - started, 3)

    capsules = capsule_rows(lane)
    write_jsonl(resolve(args.capsules_out), capsules)
    lane["outputs"] = {
        "probe": rel(probe_path),
        "capsules": rel(resolve(args.capsules_out)),
    }
    lane["summary"]["capsule_count"] = len(capsules)
    lane["summary"]["sts_row_count"] = len(capsules)

    write_json(resolve(args.out), lane)
    write_text(resolve(args.markdown_out), render_markdown(lane))
    print(json.dumps(lane, indent=2))
    return 0 if lane["trigger_state"] in {"GREEN", "YELLOW"} else 2


def run_probe(probe_path: Path) -> None:
    markdown = DEFAULT_PROBE_MD
    subprocess.run(
        [
            sys.executable,
            "scripts/pufferlib4_capability_probe.py",
            "--out",
            rel(probe_path),
            "--markdown-out",
            rel(markdown),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=90,
    )


def build_lane_report(
    probe: dict[str, Any],
    *,
    run_smoke: bool,
    skip_policy_smoke: bool,
    policy_trace_out: Path,
) -> dict[str, Any]:
    native_ok = bool(get_path(probe, ["summary", "native_backend_ok"]))
    ocean_count = int(get_path(probe, ["summary", "ocean_env_count"], 0) or 0)
    atari_enabled = bool(get_path(probe, ["summary", "atari_enabled"]))
    atari_runtime_ready = bool(get_path(probe, ["atari_policy", "atari_runtime_ready"]))
    explicit_atari_enable = bool(get_path(probe, ["atari_policy", "explicit_user_enable_present"]))
    blockers = list(probe.get("blockers") or [])
    smoke = {"attempted": False, "ok": False, "reason": "native_backend_not_ready"}
    policy_trace = {
        "row_count": 0,
        "path": rel(policy_trace_out),
        "loss_delta": 0.0,
        "accuracy_delta": 0.0,
        "rollout_reward_delta": 0.0,
        "terminal_count_delta": 0.0,
        "rollout_score_delta": 0.0,
        "backend": "",
    }
    trigger_state = "GREEN" if native_ok and ocean_count else "YELLOW"
    improvement_signal = "new_clean_evidence_produced" if trigger_state == "GREEN" else "useful_failure_residual_captured"
    fallback_smoke = {"attempted": False, "ok": False, "reason": "not_needed"}

    if native_ok and not skip_policy_smoke:
        smoke = run_native_policy_smoke(probe, policy_trace_out)
        policy_trace = smoke.get("policy_trace") if isinstance(smoke.get("policy_trace"), dict) else policy_trace
        if not smoke.get("ok"):
            trigger_state = "YELLOW"
            improvement_signal = "useful_failure_residual_captured"
            blockers.append(
                {
                    "id": "pufferlib_native_policy_smoke_failed",
                    "severity": "yellow",
                    "detail": smoke.get("reason") or smoke.get("stderr_tail") or "native backend imported but bounded policy smoke did not pass",
                }
            )
    elif not skip_policy_smoke:
        fallback_smoke = run_local_synthetic_policy_smoke(policy_trace_out)
        policy_trace = fallback_smoke.get("policy_trace") if isinstance(fallback_smoke.get("policy_trace"), dict) else policy_trace
        if fallback_smoke.get("ok"):
            trigger_state = "GREEN"
            improvement_signal = "local_synthetic_rl_policy_learning_evidence"
    elif run_smoke and native_ok:
        smoke = {"attempted": False, "ok": False, "reason": "policy_smoke_explicitly_skipped"}

    if not probe or (probe.get("trigger_state") == "RED" and not fallback_smoke.get("ok") and not smoke.get("ok")):
        trigger_state = "RED"
        improvement_signal = "useful_failure_residual_captured"
        blockers.append({"id": "pufferlib_probe_red_or_missing", "severity": "red", "detail": "Capability probe did not admit the lane."})

    native_policy_evidence = bool(smoke.get("ok") and policy_trace.get("row_count", 0) > 0)
    fallback_policy_evidence = bool(fallback_smoke.get("ok") and policy_trace.get("row_count", 0) > 0)
    return {
        "policy": "project_theseus_pufferlib4_rl_lane_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "native_backend_ready": native_ok,
            "ocean_env_count": ocean_count,
            "atari_enabled": atari_enabled,
            "atari_runtime_ready": atari_runtime_ready,
            "explicit_atari_enable_present": explicit_atari_enable,
            "run_smoke_requested": run_smoke,
            "smoke_attempted": bool(smoke.get("attempted")),
            "smoke_ok": bool(smoke.get("ok")),
            "fallback_smoke_attempted": bool(fallback_smoke.get("attempted")),
            "fallback_smoke_ok": bool(fallback_smoke.get("ok")),
            "native_policy_learning_evidence": native_policy_evidence,
            "fallback_policy_learning_evidence": fallback_policy_evidence,
            "policy_learning_evidence": bool(native_policy_evidence or fallback_policy_evidence),
            "policy_learning_backend": policy_trace.get("backend") or ("puffer_native" if native_policy_evidence else "local_synthetic_rl" if fallback_policy_evidence else ""),
            "policy_train_row_count": int(policy_trace.get("row_count") or 0),
            "policy_trace": policy_trace.get("path"),
            "policy_loss_delta": policy_trace.get("loss_delta"),
            "policy_accuracy_delta": policy_trace.get("accuracy_delta"),
            "policy_rollout_reward_delta": policy_trace.get("rollout_reward_delta"),
            "policy_terminal_count_delta": policy_trace.get("terminal_count_delta"),
            "policy_rollout_score_delta": policy_trace.get("rollout_score_delta"),
            "improvement_signal": improvement_signal,
        },
        "blockers": blockers,
        "smoke": smoke,
        "fallback_smoke": fallback_smoke,
        "admission": {
            "allowed_training_targets": ["puffer_ocean", "local_board_game_rl", "local_synthetic_rl"] if native_ok else ["local_board_game_rl", "local_synthetic_rl"],
            "atari_status": atari_status(atari_enabled, atari_runtime_ready, explicit_atari_enable),
            "commercial_rom_fetching": "forbidden",
            "public_benchmark_data": "calibration_only",
        },
        "fallback_sources": [
            "board_game_rl",
            "long_horizon_tool_use",
            "symliquid_gridworld",
            "legacy_rl_seeded_smoke",
        ],
        "transfer_targets": [
            "legal_action_masks",
            "state_memory",
            "sparse_reward_credit_assignment",
            "branching_plan_selection",
            "reset_step_contracts",
            "rollout_replay",
            "policy_value_trace",
            "repair_after_loss",
        ],
    }


def run_native_policy_smoke(probe: dict[str, Any], policy_trace_out: Path) -> dict[str, Any]:
    python_path = resolve(str(get_path(probe, ["summary", "puffer_python"], ""))) if get_path(probe, ["summary", "puffer_python"], "") else PUFFER_VENV_PY
    if not python_path.exists():
        python_path = Path(sys.executable)
    policy_trace_out.parent.mkdir(parents=True, exist_ok=True)
    D_TMP.mkdir(parents=True, exist_ok=True)
    code = r"""
import ctypes
import importlib, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pufferlib import models

trace = Path(os.environ["THESEUS_PUFFER_POLICY_TRACE"])
trace.parent.mkdir(parents=True, exist_ok=True)
backend = importlib.import_module("pufferlib._C")

torch.manual_seed(7)
random.seed(7)
np.random.seed(7)
args = {
    "vec": {"total_agents": 64, "num_buffers": 1},
    "env": {
        "cart_mass": 1.0,
        "pole_mass": 0.1,
        "pole_length": 0.5,
        "gravity": 9.8,
        "force_mag": 10.0,
        "dt": 0.02,
        "continuous": 0,
    },
}
vec = backend.create_vec(args, 0)
vec.reset()
obs_size = int(getattr(vec, "obs_size", 4))
num_agents = int(getattr(vec, "total_agents", 64))
num_atns = int(getattr(vec, "num_atns", 1))
obs_buf = np.ctypeslib.as_array((ctypes.c_float * (num_agents * obs_size)).from_address(vec.obs_ptr)).reshape(num_agents, obs_size)
actions_buf = np.ctypeslib.as_array((ctypes.c_float * (num_agents * num_atns)).from_address(vec.actions_ptr)).reshape(num_agents, num_atns)
rewards_buf = np.ctypeslib.as_array((ctypes.c_float * num_agents).from_address(vec.rewards_ptr))
terminals_buf = np.ctypeslib.as_array((ctypes.c_float * num_agents).from_address(vec.terminals_ptr))
hidden = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = models.Policy(models.DefaultEncoder(obs_size, hidden), models.DefaultDecoder([2], hidden), models.MLP(hidden, num_layers=1)).to(device)
opt = torch.optim.Adam(policy.parameters(), lr=0.03)

def labels_for(obs_tensor):
    flat = obs_tensor[:, 0, :]
    signal = flat[:, 2] + 0.15 * flat[:, 3] + 0.01 * flat[:, 1]
    return (signal > 0).long()

def tensor_obs():
    return torch.from_numpy(obs_buf.copy()).reshape(num_agents, 1, obs_size).float().to(device, non_blocking=True)

def policy_outputs(obs):
    logits, values = policy(obs)
    return logits.reshape(-1, 2), values.reshape(-1)

def score():
    obs = tensor_obs()
    labels = labels_for(obs)
    logits, values = policy_outputs(obs)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return float(loss.item()), float(acc)

def rollout(steps):
    total_reward = 0.0
    terminal_count = 0.0
    for _ in range(steps):
        obs = torch.from_numpy(obs_buf.copy()).reshape(num_agents, 1, obs_size).float().to(device)
        with torch.no_grad():
            logits, values = policy(obs)
            action = logits.reshape(num_agents, 2).argmax(dim=1).float().detach().cpu().numpy()
        actions_buf[:, 0] = action
        vec.cpu_step(int(actions_buf.ctypes.data))
        total_reward += float(rewards_buf.sum())
        terminal_count += float(terminals_buf.sum())
    logs = dict(vec.log())
    return {
        "mean_reward_per_agent_step": total_reward / max(1, steps * num_agents),
        "terminal_count": terminal_count,
        "log_score": float(logs.get("score", 0.0) or 0.0),
        "log_n": float(logs.get("n", 0.0) or 0.0),
    }

def collect_replay(steps, gamma=0.97):
    obs_rows = []
    action_rows = []
    reward_rows = []
    done_rows = []
    for _ in range(steps):
        obs = tensor_obs()
        with torch.no_grad():
            logits, values = policy_outputs(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        obs_rows.append(obs.detach())
        action_rows.append(action.detach())
        actions_buf[:, 0] = action.detach().cpu().numpy().astype(np.float32)
        vec.cpu_step(int(actions_buf.ctypes.data))
        reward_rows.append(torch.from_numpy(rewards_buf.copy()).float().to(device, non_blocking=True))
        done_rows.append(torch.from_numpy(terminals_buf.copy()).float().to(device, non_blocking=True))
    with torch.no_grad():
        _, next_value = policy_outputs(tensor_obs())
    returns = []
    running_return = next_value.detach()
    for reward, done in zip(reversed(reward_rows), reversed(done_rows)):
        running_return = reward + gamma * running_return * (1.0 - done)
        returns.append(running_return.detach())
    returns.reverse()
    return (
        torch.cat(obs_rows, dim=0),
        torch.cat(action_rows, dim=0).long(),
        torch.cat(returns, dim=0),
        torch.stack(reward_rows).mean().item(),
    )

rollout_before = rollout(128)
vec.reset()
loss_before, acc_before = score()
rows = []
for update in range(24):
    obs, actions, returns, replay_reward_mean = collect_replay(16)
    logits, values = policy_outputs(obs)
    dist = torch.distributions.Categorical(logits=logits)
    log_prob = dist.log_prob(actions)
    advantages = returns - values.detach()
    advantage_std = advantages.std().clamp_min(1e-6)
    advantages = (advantages - advantages.mean()) / advantage_std
    policy_loss = -(log_prob * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = dist.entropy().mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    rows.append({
        "update": update,
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "advantage_std": float(advantage_std.item()),
        "native_reward_mean": float(replay_reward_mean),
        "torch_device": str(device),
    })
loss_after, acc_after = score()
vec.reset()
rollout_after = rollout(128)
vec.close()
with trace.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
print(json.dumps({
    "backend_env_name": getattr(backend, "env_name", ""),
    "precision_bytes": getattr(backend, "precision_bytes", None),
    "native_vec_total_agents": num_agents,
    "native_vec_obs_size": obs_size,
    "native_vec_num_atns": num_atns,
    "native_rollout_before": rollout_before,
    "native_rollout_after": rollout_after,
    "loss_before": loss_before,
    "loss_after": loss_after,
    "loss_delta": loss_before - loss_after,
    "accuracy_before": acc_before,
    "accuracy_after": acc_after,
    "accuracy_delta": acc_after - acc_before,
    "actor_critic_updates": len(rows),
    "advantage_value_learning": True,
    "last_policy_loss": rows[-1]["policy_loss"] if rows else None,
    "last_value_loss": rows[-1]["value_loss"] if rows else None,
    "last_entropy": rows[-1]["entropy"] if rows else None,
    "torch_device": str(device),
    "torch_cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    "row_count": len(rows),
    "trace": str(trace),
}, sort_keys=True))
"""
    env = os.environ.copy()
    env["TEMP"] = str(D_TMP)
    env["TMP"] = str(D_TMP)
    env["PIP_CACHE_DIR"] = str(DEFAULT_CACHE_DIR / "pip-cache")
    env["THESEUS_PUFFER_POLICY_TRACE"] = str(policy_trace_out)
    current = env.get("PYTHONPATH", "")
    paths = [str(VENDOR_PUFFER)]
    if current:
        paths.append(current)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [str(python_path), "-c", code],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "attempted": True,
            "ok": False,
            "reason": "native_policy_smoke_timeout",
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": tail(exc.stdout or "", 1000),
            "stderr_tail": tail(exc.stderr or "", 1000),
        }
    payload = parse_json_output(result.stdout)
    rollout_before = payload.get("native_rollout_before") if isinstance(payload, dict) else {}
    rollout_after = payload.get("native_rollout_after") if isinstance(payload, dict) else {}
    reward_delta = (
        floatish(rollout_after.get("mean_reward_per_agent_step"))
        - floatish(rollout_before.get("mean_reward_per_agent_step"))
        if isinstance(rollout_before, dict) and isinstance(rollout_after, dict)
        else 0.0
    )
    terminal_delta = (
        floatish(rollout_after.get("terminal_count"))
        - floatish(rollout_before.get("terminal_count"))
        if isinstance(rollout_before, dict) and isinstance(rollout_after, dict)
        else 0.0
    )
    score_delta = (
        floatish(rollout_after.get("log_score"))
        - floatish(rollout_before.get("log_score"))
        if isinstance(rollout_before, dict) and isinstance(rollout_after, dict)
        else 0.0
    )
    ok = (
        result.returncode == 0
        and isinstance(payload, dict)
        and int(payload.get("row_count") or 0) >= 16
        and bool(payload.get("advantage_value_learning"))
        and "cuda" in str(payload.get("torch_device") or "").lower()
        and (reward_delta >= 0.02 or terminal_delta <= -32.0 or score_delta > 0.0)
    )
    return {
        "attempted": True,
        "ok": ok,
        "reason": "native_policy_learning_smoke_passed" if ok else "native_policy_learning_smoke_failed",
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "returncode": result.returncode,
        "policy_trace": {
            "path": rel(policy_trace_out),
            "row_count": int(payload.get("row_count") or 0) if isinstance(payload, dict) else 0,
            "backend": "puffer_native",
            "loss_delta": round(float(payload.get("loss_delta") or 0.0), 6) if isinstance(payload, dict) else 0.0,
            "accuracy_delta": round(float(payload.get("accuracy_delta") or 0.0), 6) if isinstance(payload, dict) else 0.0,
            "backend_env_name": payload.get("backend_env_name") if isinstance(payload, dict) else "",
            "precision_bytes": payload.get("precision_bytes") if isinstance(payload, dict) else None,
            "native_vec_total_agents": payload.get("native_vec_total_agents") if isinstance(payload, dict) else None,
            "native_vec_obs_size": payload.get("native_vec_obs_size") if isinstance(payload, dict) else None,
            "actor_critic_updates": payload.get("actor_critic_updates") if isinstance(payload, dict) else None,
            "advantage_value_learning": payload.get("advantage_value_learning") if isinstance(payload, dict) else False,
            "last_policy_loss": payload.get("last_policy_loss") if isinstance(payload, dict) else None,
            "last_value_loss": payload.get("last_value_loss") if isinstance(payload, dict) else None,
            "last_entropy": payload.get("last_entropy") if isinstance(payload, dict) else None,
            "rollout_reward_delta": round(reward_delta, 6),
            "terminal_count_delta": round(terminal_delta, 6),
            "rollout_score_delta": round(score_delta, 6),
            "torch_device": payload.get("torch_device") if isinstance(payload, dict) else "",
            "torch_cuda_device_name": payload.get("torch_cuda_device_name") if isinstance(payload, dict) else "",
            "native_rollout_before": payload.get("native_rollout_before") if isinstance(payload, dict) else None,
            "native_rollout_after": payload.get("native_rollout_after") if isinstance(payload, dict) else None,
        },
        "stdout_tail": tail(result.stdout, 1000),
        "stderr_tail": tail(result.stderr, 1000),
    }


def run_local_synthetic_policy_smoke(policy_trace_out: Path) -> dict[str, Any]:
    """Train a small private tabular policy when Puffer native is unavailable.

    This is intentionally not presented as Puffer/Ocean readiness. It gives the
    Hive a real policy-learning trace for legal-action masking, sparse reward,
    state memory, and delayed credit assignment on machines that cannot build
    the native backend yet.
    """

    started = time.perf_counter()
    rng = random.Random(17)
    policy_trace_out.parent.mkdir(parents=True, exist_ok=True)
    width = 5
    height = 5
    start = (0, 0, 0)
    key_pos = (2, 0)
    goal_pos = (4, 4)
    walls = {(1, 1), (1, 2), (3, 2), (3, 3)}
    actions = {
        0: (0, -1, "up"),
        1: (1, 0, "right"),
        2: (0, 1, "down"),
        3: (-1, 0, "left"),
    }

    def legal_actions(state: tuple[int, int, int]) -> list[int]:
        x, y, _has_key = state
        out: list[int] = []
        for action, (dx, dy, _name) in actions.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                out.append(action)
        return out

    def step_state(state: tuple[int, int, int], action: int) -> tuple[tuple[int, int, int], float, bool]:
        x, y, has_key = state
        dx, dy, _name = actions[action]
        nx, ny = x + dx, y + dy
        reward = -0.01
        if not (0 <= nx < width and 0 <= ny < height) or (nx, ny) in walls:
            nx, ny = x, y
            reward -= 0.05
        if (nx, ny) == key_pos and not has_key:
            has_key = 1
            reward += 0.25
        done = (nx, ny) == goal_pos and bool(has_key)
        if done:
            reward += 1.0
        return (nx, ny, has_key), reward, done

    def greedy_action(q: dict[tuple[tuple[int, int, int], int], float], state: tuple[int, int, int]) -> int:
        choices = legal_actions(state)
        return max(choices, key=lambda action: (q.get((state, action), 0.0), -action))

    def run_episode(q: dict[tuple[tuple[int, int, int], int], float], epsilon: float, train: bool) -> dict[str, Any]:
        state = start
        total_reward = 0.0
        updates = 0
        key_collected = False
        path: list[dict[str, Any]] = []
        for t in range(40):
            choices = legal_actions(state)
            if train and rng.random() < epsilon:
                action = rng.choice(choices)
            else:
                action = greedy_action(q, state)
            next_state, reward, done = step_state(state, action)
            total_reward += reward
            key_collected = key_collected or bool(next_state[2])
            if train:
                next_choices = legal_actions(next_state)
                best_next = max((q.get((next_state, item), 0.0) for item in next_choices), default=0.0)
                old = q.get((state, action), 0.0)
                q[(state, action)] = old + 0.35 * (reward + 0.94 * best_next - old)
                updates += 1
            if t < 8:
                path.append(
                    {
                        "state": state,
                        "action": actions[action][2],
                        "legal_action_count": len(choices),
                        "reward": round(reward, 4),
                        "next_state": next_state,
                    }
                )
            state = next_state
            if done:
                return {
                    "success": True,
                    "steps": t + 1,
                    "reward": round(total_reward, 6),
                    "updates": updates,
                    "key_collected": key_collected,
                    "path_excerpt": path,
                }
        return {
            "success": False,
            "steps": 40,
            "reward": round(total_reward, 6),
            "updates": updates,
            "key_collected": key_collected,
            "path_excerpt": path,
        }

    def evaluate(q: dict[tuple[tuple[int, int, int], int], float], episodes: int) -> dict[str, Any]:
        rows = [run_episode(q, epsilon=0.0, train=False) for _ in range(episodes)]
        return {
            "success_rate": sum(1 for row in rows if row["success"]) / max(1, len(rows)),
            "mean_reward": sum(float(row["reward"]) for row in rows) / max(1, len(rows)),
            "mean_steps": sum(int(row["steps"]) for row in rows) / max(1, len(rows)),
            "sample": rows[0] if rows else {},
        }

    q_values: dict[tuple[tuple[int, int, int], int], float] = {}
    before = evaluate(q_values, 32)
    rows: list[dict[str, Any]] = []
    for episode in range(192):
        epsilon = max(0.05, 0.85 * (1.0 - episode / 192))
        result = run_episode(q_values, epsilon=epsilon, train=True)
        if episode % 4 == 0 or episode >= 184:
            rows.append(
                {
                    "episode": episode,
                    "epsilon": round(epsilon, 4),
                    "success": result["success"],
                    "steps": result["steps"],
                    "reward": result["reward"],
                    "updates": result["updates"],
                    "q_state_action_count": len(q_values),
                    "legal_action_masked": True,
                    "sparse_reward_credit_assignment": True,
                    "state_memory_required": True,
                }
            )
    after = evaluate(q_values, 32)
    with policy_trace_out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    accuracy_delta = float(after["success_rate"]) - float(before["success_rate"])
    reward_delta = float(after["mean_reward"]) - float(before["mean_reward"])
    ok = len(rows) >= 32 and accuracy_delta >= 0.75 and reward_delta > 0.5
    return {
        "attempted": True,
        "ok": ok,
        "reason": "local_synthetic_policy_learning_passed" if ok else "local_synthetic_policy_learning_failed",
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "policy_trace": {
            "path": rel(policy_trace_out),
            "row_count": len(rows),
            "backend": "local_synthetic_rl",
            "loss_delta": 0.0,
            "accuracy_delta": round(accuracy_delta, 6),
            "rollout_reward_delta": round(reward_delta, 6),
            "terminal_count_delta": round((after["success_rate"] - before["success_rate"]) * 32, 6),
            "rollout_score_delta": round(reward_delta, 6),
            "task": "private_keydoor_gridworld",
            "legal_action_masking": True,
            "state_memory_required": True,
            "sparse_reward_credit_assignment": True,
            "evaluation_before": before,
            "evaluation_after": after,
        },
    }


def capsule_rows(lane: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    blocker_ids = [str(row.get("id") or "") for row in lane.get("blockers", []) if isinstance(row, dict)]
    for skill in lane.get("transfer_targets", []):
        rows.append(
            {
                "created_utc": now(),
                "policy": "project_theseus_pufferlib4_rl_sts_capsule_v1",
                "source_lane": "pufferlib4_rl",
                "skill": skill,
                "quality": "admitted_policy_trace" if get_path(lane, ["summary", "policy_learning_evidence"]) else "diagnostic",
                "training_eligible": bool(get_path(lane, ["summary", "policy_learning_evidence"])),
                "policy_learning_backend": get_path(lane, ["summary", "policy_learning_backend"]),
                "metadata_only": True,
                "rom_free": True,
                "blockers": blocker_ids,
                "transfer_receivers": ["code_decoder_contracts", "repo_repair", "long_horizon_tool_use", "conversation_while_working"],
            }
        )
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PufferLib 4 RL Lane",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- native_backend_ready: `{get_path(report, ['summary', 'native_backend_ready'])}`",
        f"- ocean_env_count: `{get_path(report, ['summary', 'ocean_env_count'])}`",
        f"- atari_enabled: `{get_path(report, ['summary', 'atari_enabled'])}`",
        f"- atari_status: `{get_path(report, ['admission', 'atari_status'])}`",
        f"- improvement_signal: `{get_path(report, ['summary', 'improvement_signal'])}`",
        f"- native_policy_learning_evidence: `{get_path(report, ['summary', 'native_policy_learning_evidence'])}`",
        f"- fallback_policy_learning_evidence: `{get_path(report, ['summary', 'fallback_policy_learning_evidence'])}`",
        f"- policy_learning_backend: `{get_path(report, ['summary', 'policy_learning_backend'])}`",
        f"- policy_train_row_count: `{get_path(report, ['summary', 'policy_train_row_count'])}`",
        f"- capsule_count: `{get_path(report, ['summary', 'capsule_count'])}`",
        "",
        "## Blockers",
    ]
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    if blockers:
        lines.extend(f"- `{row.get('id')}`: {row.get('detail')}" for row in blockers if isinstance(row, dict))
    else:
        lines.append("- none")
    lines.extend(["", "## Transfer Targets"])
    lines.extend(f"- `{item}`" for item in report.get("transfer_targets", []))
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_json_output(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    for line in reversed(text.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return default
        if cur is None:
            return default
    return cur


def tail(text: Any, chars: int = 1000) -> str:
    return str(text or "")[-chars:]


def floatish(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def atari_status(enabled: bool, runtime_ready: bool, explicit_enable: bool) -> str:
    if enabled:
        return "enabled_with_explicit_user_supplied_assets"
    if runtime_ready and not explicit_enable:
        return "runtime_ready_but_disabled_until_configs_allow_user_supplied_atari_flag"
    return "disabled_until_ale_legal_assets_and_explicit_flag"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
