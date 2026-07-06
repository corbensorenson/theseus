"""Octopus-routed autonomous goal runner.

This is the operational bridge between the RMI/ORA papers and the local
automation loop. A goal enters through the resident head, gets routed to arms,
receives a resource envelope, runs known low-risk local tools when permitted,
and escalates to the teacher only when local routing cannot produce a bounded
plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import coherence_delirium_gate  # noqa: E402
import personality_context_builder  # noqa: E402
REPORTS = ROOT / "reports"
DEFAULT_POLICY = ROOT / "configs" / "autonomy_policy.json"
DEFAULT_ARM_REGISTRY = REPORTS / "arm_registry.json"
DEFAULT_RESOURCE = REPORTS / "resource_governor.json"
DEFAULT_OUT = REPORTS / "autonomous_goal_last.json"
DEFAULT_LEDGER = REPORTS / "autonomous_goal_ledger.jsonl"
DEFAULT_ROUTING_TRACE = REPORTS / "routing_memory_real_traces.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--profile", default="inner_loop")
    parser.add_argument("--risk", default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--allow-network-fetch", action="store_true")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--arm-registry", default=str(DEFAULT_ARM_REGISTRY.relative_to(ROOT)))
    parser.add_argument("--resource", default=str(DEFAULT_RESOURCE.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    arms = (read_json(ROOT / args.arm_registry).get("arms") or [])
    risk = infer_risk(args.goal) if args.risk == "auto" else args.risk
    resource = ensure_resource_report(args.profile)
    coherence_gate = ensure_coherence_gate()
    runtime_enforcement = ensure_runtime_enforcement()
    personality_context = personality_context_builder.build_context(prompt=args.goal, task="autonomous_goal")
    plan = build_plan(
        args.goal,
        args.profile,
        risk,
        arms,
        resource,
        coherence_gate,
        runtime_enforcement,
        args.allow_network_fetch,
        args.execute,
        personality_context,
    )
    commands = []
    local_insufficient = plan["local_confidence"] < 0.35 or not plan["commands"]
    if plan.get("governance_blocked"):
        local_insufficient = False
        plan["blocked_reason"] = plan.get("blocked_reason") or "governance_blocked"
    if plan["resource_decision"].get("can_run_requested_profile") is False and plan["requires_training"]:
        local_insufficient = False
        plan["blocked_reason"] = "resource_governor_throttle"
    if args.execute and not local_insufficient:
        for command in plan["commands"]:
            commands.append(run_step(command, timeout=timeout_for_command(command, policy), allow_failure=command_allow_failure(command)))
    elif plan["commands"]:
        commands = [skipped_step(command, "planned_without_execute") for command in plan["commands"]]

    teacher_result = None
    teacher_needed = local_insufficient or plan.get("teacher_recommended", False) or plan.get("review_required", False)
    if teacher_needed:
        teacher_result = call_teacher(args.goal, plan, args.allow_teacher)
        commands.append(teacher_result["step"])

    outcome = goal_outcome(commands, local_insufficient, plan)
    report = {
        "policy": "sparkstream_autonomous_goal_runner_v0",
        "goal_id": goal_id(args.goal),
        "created_utc": now(),
        "goal": args.goal,
        "profile": args.profile,
        "risk": risk,
        "execute": args.execute,
        "allow_teacher": args.allow_teacher,
        "allow_network_fetch": args.allow_network_fetch,
        "selected_arms": plan["selected_arms"],
        "permission_envelopes": plan["permission_envelopes"],
        "resource": resource,
        "personality_context": compact_personality_context(personality_context),
        "legacy_port_runtime_enforcement": compact_runtime_enforcement(runtime_enforcement),
        "plan": plan,
        "commands": commands,
        "teacher_needed": teacher_needed,
        "teacher_used": bool(args.allow_teacher and teacher_needed),
        "teacher": teacher_result.get("result") if teacher_result else None,
        "outcome": outcome,
    }
    write_json(ROOT / args.out, report)
    append_jsonl(DEFAULT_LEDGER, compact_goal_ledger(report))
    append_jsonl(DEFAULT_ROUTING_TRACE, routing_trace(report))
    print(json.dumps(report, indent=2))
    return 0 if outcome["ok"] else 1


def ensure_resource_report(profile: str) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            "scripts/resource_governor.py",
            "--profile",
            profile,
            "--out",
            "reports/resource_governor.json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return read_json(DEFAULT_RESOURCE)


def ensure_coherence_gate() -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            "scripts/coherence_delirium_gate.py",
            "--out",
            "reports/coherence_delirium_gate.json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    gate = read_json(REPORTS / "coherence_delirium_gate.json")
    return gate or coherence_delirium_gate.load_gate()


def ensure_runtime_enforcement() -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            "scripts/legacy_port_runtime_enforcer.py",
            "--out",
            "reports/legacy_port_runtime_enforcement.json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )
    return read_json(REPORTS / "legacy_port_runtime_enforcement.json")


def build_plan(
    goal: str,
    profile: str,
    risk: str,
    arms: list[dict[str, Any]],
    resource: dict[str, Any],
    coherence_gate: dict[str, Any],
    runtime_enforcement: dict[str, Any],
    allow_network_fetch: bool,
    execute_requested: bool,
    personality_context: dict[str, Any],
) -> dict[str, Any]:
    selected = route_arms(goal, risk, arms)
    permission_envelopes = {
        arm["arm_name"]: permission_envelope(arm, risk)
        for arm in selected
    }
    commands = commands_for_goal(goal, profile, allow_network_fetch, execute_requested)
    requires_training = any(command_requires_training(command) for command in commands)
    resource_decision = resource.get("decision") or {}
    if requires_training and resource_decision.get("can_run_requested_profile") is False:
        commands = [command for command in commands if not command_requires_training(command)]
    risky_capability_goal = risk in {"high", "critical"} or any(
        token in goal.lower()
        for token in ["candidate", "architecture", "self edit", "capability", "deploy", "long run"]
    )
    long_run_goal = any(token in goal.lower() for token in ["long run", "launch", "deploy", "candidate", "self edit"])
    coherence_blocked = bool(
        risky_capability_goal and execute_requested and not coherence_gate.get("allows_capability_expansion", False)
    )
    runtime_blocked = bool(
        risky_capability_goal
        and execute_requested
        and (
            not runtime_enforcement.get("ready_for_bounded_autonomy", False)
            or (long_run_goal and not runtime_enforcement.get("ready_for_long_autonomy", False))
        )
    )
    governance_blocked = coherence_blocked or runtime_blocked
    if governance_blocked:
        commands = []
    confidence = local_confidence(goal, selected, commands)
    return {
        "routing_strategy": "keyword_weighted_arm_cards_with_resource_gate_v0",
        "selected_arms": [arm["arm_name"] for arm in selected],
        "permission_envelopes": permission_envelopes,
        "commands": commands,
        "requires_training": requires_training,
        "resource_decision": resource_decision,
        "coherence_delirium_gate": compact_coherence_gate(coherence_gate),
        "legacy_port_runtime_enforcement": compact_runtime_enforcement(runtime_enforcement),
        "governance_blocked": governance_blocked,
        "blocked_reason": (
            "coherence_delirium_gate_blocked"
            if coherence_blocked
            else "legacy_port_runtime_enforcement_blocked"
            if runtime_blocked
            else ""
        ),
        "review_required": bool(coherence_gate.get("review_required")) and risky_capability_goal,
        "local_confidence": confidence,
        "teacher_recommended": confidence < 0.55 and not commands,
        "personality_context": compact_personality_context(personality_context),
        "reality_orientation": personality_context_builder.orientation_answer(goal, personality_context)
        if personality_context.get("status") == "ready"
        else "",
        "efficiency_notes": [
            "use smoke/profile gates before longer training",
            "prefer Rust/CUDA for hot loops",
            "avoid network and teacher unless explicit policy allows them",
            "append real routing traces after every goal",
        ],
    }


def route_arms(goal: str, risk: str, arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    goal_l = goal.lower()
    scored = []
    for arm in arms:
        score = 0
        for keyword in arm.get("routing_keywords", []):
            if str(keyword).lower() in goal_l:
                score += max(1, len(str(keyword).split()))
        if arm.get("arm_name") == "head_router":
            score += 100
        if risk in {"high", "critical"} and arm.get("arm_name") == "safety_reflex_arm":
            score += 50
        if any(token in goal_l for token in ["efficient", "resource", "gpu", "vram", "cuda", "rust"]):
            if arm.get("arm_name") == "rust_cuda_systems_arm":
                score += 20
        if score > 0:
            scored.append((score, arm))
    selected = [arm for _, arm in sorted(scored, key=lambda item: item[0], reverse=True)]
    if not any(arm.get("arm_name") == "residual_governance_arm" for arm in selected) and any(
        token in goal_l for token in ["improve", "cannot", "stuck", "wall", "failure", "residual"]
    ):
        match = next((arm for arm in arms if arm.get("arm_name") == "residual_governance_arm"), None)
        if match:
            selected.append(match)
    return selected[:6]


def commands_for_goal(
    goal: str,
    profile: str,
    allow_network_fetch: bool,
    execute_requested: bool,
) -> list[list[str]]:
    goal_l = goal.lower()
    commands: list[list[str]] = []
    if any(token in goal_l for token in ["resource", "efficient", "gpu", "vram", "budget"]):
        commands.append([sys.executable, "scripts/resource_governor.py", "--profile", profile, "--out", "reports/resource_governor.json"])
    if any(token in goal_l for token in ["refresh", "inventory", "data", "benchmark list", "what benchmarks"]):
        commands.extend(
            [
                [sys.executable, "scripts/benchmark_seeker.py", "--refresh-local", "--out", "reports/benchmark_seeker_registry.json"],
                [sys.executable, "scripts/training_data_inventory.py", "--out", "reports/training_data_inventory.json"],
                [sys.executable, "scripts/rl_benchmark_registry.py", "--refresh-local", "--out", "reports/rl_benchmark_registry.json"],
            ]
        )
    if "router" in goal_l or "arm" in goal_l or "octopus" in goal_l:
        commands.append([sys.executable, "scripts/octopus_router.py", "--out", "reports/octopus_router_report.json"])
        commands.append([sys.executable, "scripts/train_octopus_router_head.py", "--out", "reports/octopus_router_head_report.json"])
    if any(token in goal_l for token in ["preflight", "ready for training", "training ready"]):
        commands.append([sys.executable, "scripts/training_preflight.py", "--out", "reports/training_preflight_report.json"])
    if any(token in goal_l for token in ["improve", "ratchet", "frontier", "self learn", "self-learning", "train"]):
        command = [sys.executable, "scripts/autonomy_cycle.py", "--profile", profile, "--out", "reports/autonomy_cycle_last.json"]
        if execute_requested:
            command.append("--execute")
        commands.append(command)
    if any(token in goal_l for token in ["checkpoint", "snapshot", "version"]):
        commands.append(
            [
                sys.executable,
                "scripts/checkpoint_registry.py",
                "create",
                "--label",
                "autonomous_goal_checkpoint",
                "--reason",
                "autonomous_goal_runner",
                "--profile",
                profile,
                "--status",
                "observed",
                "--out",
                "reports/checkpoint_last.json",
            ]
        )
    if allow_network_fetch and any(token in goal_l for token in ["discover", "find benchmark", "new benchmark", "rl source"]):
        commands.append(
            [
                sys.executable,
                "scripts/rl_benchmark_registry.py",
                "--refresh-local",
                "--allow-network-discovery",
                "--discover-query",
                "open source reinforcement learning benchmark",
                "--discover-limit",
                "10",
                "--out",
                "reports/rl_benchmark_registry.json",
            ]
        )
    return dedupe_commands(commands)


def permission_envelope(arm: dict[str, Any], risk: str) -> dict[str, Any]:
    boundary = arm.get("permission_boundary") or {}
    side_effects = list(boundary.get("side_effects") or [])
    if risk in {"high", "critical"} and arm.get("arm_name") != "safety_reflex_arm":
        side_effects = [item for item in side_effects if item != "write_generated_benchmarks"]
        side_effects.append("dry_run_only")
    if risk == "critical":
        side_effects.append("human_approval_required")
    return {
        "memory": boundary.get("memory", []),
        "tools": boundary.get("tools", "allowlisted_only"),
        "side_effects": sorted(set(side_effects)),
        "runtime_tier": arm.get("runtime_tier"),
        "risk": risk,
        "network": boundary.get("network", "disabled_for_inner_loop"),
        "external_inference": boundary.get("external_inference", "forbidden"),
        "budget": arm.get("cost_profile", {}),
    }


def infer_risk(goal: str) -> str:
    goal_l = goal.lower()
    if any(token in goal_l for token in ["production", "deploy", "delete", "overwrite", "financial", "legal", "security"]):
        return "critical"
    if any(token in goal_l for token in ["network", "download", "teacher", "architecture", "long run", "candidate"]):
        return "high"
    if any(token in goal_l for token in ["train", "execute", "write", "checkpoint", "ratchet"]):
        return "medium"
    return "low"


def local_confidence(goal: str, selected: list[dict[str, Any]], commands: list[list[str]]) -> float:
    if not selected:
        return 0.0
    score = 0.25 + min(0.45, 0.08 * len(selected))
    if commands:
        score += 0.25
    if any("teacher" in word.lower() for word in goal.split()):
        score -= 0.15
    return round(max(0.0, min(0.95, score)), 4)


def run_step(command: list[str], *, timeout: int, allow_failure: bool) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {
            "name": Path(command[1]).stem if len(command) > 1 else command[0],
            "command": command,
            "returncode": result.returncode,
            "allow_failure": allow_failure,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-3000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": Path(command[1]).stem if len(command) > 1 else command[0],
            "command": command,
            "returncode": 124,
            "allow_failure": allow_failure,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": (exc.stdout or "")[-3000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-3000:] if isinstance(exc.stderr, str) else "",
            "error": "timeout",
        }


def skipped_step(command: list[str], reason: str) -> dict[str, Any]:
    return {
        "name": Path(command[1]).stem if len(command) > 1 else command[0],
        "command": command,
        "returncode": 0,
        "allow_failure": False,
        "runtime_ms": 0,
        "skipped": True,
        "reason": reason,
    }


def call_teacher(goal: str, plan: dict[str, Any], allow_teacher: bool) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/teacher_oracle.py",
        "--reason",
        "architecture_wall",
        "--mode",
        "proposal",
        "--local-evidence",
        f"goal={goal}",
        f"selected_arms={','.join(plan.get('selected_arms') or [])}",
        f"local_confidence={plan.get('local_confidence')}",
        f"blocked_reason={plan.get('blocked_reason', '')}",
        f"personality_context={json.dumps(plan.get('personality_context') or {}, sort_keys=True)}",
        f"reality_orientation={plan.get('reality_orientation', '')}",
        "--out",
        "reports/teacher_oracle_last.json",
    ]
    if allow_teacher:
        command.append("--allow-teacher")
    step = run_step(command, timeout=1800, allow_failure=True)
    return {"step": step, "result": read_json(REPORTS / "teacher_oracle_last.json")}


def goal_outcome(commands: list[dict[str, Any]], local_insufficient: bool, plan: dict[str, Any]) -> dict[str, Any]:
    hard_failures = [cmd for cmd in commands if cmd.get("returncode", 0) != 0 and not cmd.get("allow_failure")]
    return {
        "ok": not hard_failures and not plan.get("governance_blocked", False),
        "local_insufficient": local_insufficient,
        "hard_failures": len(hard_failures),
        "blocked_reason": plan.get("blocked_reason", ""),
        "verification": "commands_completed_or_planned; route_trace_appended",
    }


def timeout_for_command(command: list[str], policy: dict[str, Any]) -> int:
    text = " ".join(command)
    if command_requires_training(command) or "autonomy_cycle.py" in text:
        return int((policy.get("command_timeouts_seconds") or {}).get("inner_loop", 2700))
    return int((policy.get("command_timeouts_seconds") or {}).get("maintenance", 1800))


def command_requires_training(command: list[str]) -> bool:
    text = " ".join(command)
    if "run_training_ratchet_profile.py" in text:
        return True
    return "autonomy_cycle.py" in text and "--execute" in command


def command_allow_failure(command: list[str]) -> bool:
    text = " ".join(command)
    return "candidate_promotion_gate.py" in text or "teacher_oracle.py" in text


def compact_goal_ledger(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal_id": report.get("goal_id"),
        "created_utc": report.get("created_utc"),
        "goal": report.get("goal"),
        "profile": report.get("profile"),
        "risk": report.get("risk"),
        "execute": report.get("execute"),
        "selected_arms": report.get("selected_arms"),
        "personality_context": get_path(report, ["personality_context", "summary"], {}),
        "legacy_port_runtime_enforcement": get_path(report, ["legacy_port_runtime_enforcement"], {}),
        "teacher_needed": report.get("teacher_needed"),
        "teacher_used": report.get("teacher_used"),
        "ok": get_path(report, ["outcome", "ok"], False),
        "efficiency_score": get_path(report, ["resource", "efficiency", "score"], None),
        "review_step_count": goal_review_step_count(report),
        "review_step_basis": "selected_arms_permission_resource_outcome_teacher",
        "maintenance_mode": maintenance_mode_from_report(report),
        "maintenance_mode_basis": "report_goal_or_object_only_default",
        "human_edit_minutes": None,
        "human_edit_minutes_measured": False,
    }


def routing_trace(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": report.get("goal_id"),
        "created_utc": report.get("created_utc"),
        "source": "autonomous_goal_runner",
        "task": report.get("goal"),
        "risk": report.get("risk"),
        "selected_arms": report.get("selected_arms"),
        "permission_envelopes": report.get("permission_envelopes"),
        "personality_context": report.get("personality_context"),
        "reality_orientation": get_path(report, ["plan", "reality_orientation"], ""),
        "outcome": report.get("outcome"),
        "resource_summary": {
            "efficiency_score": get_path(report, ["resource", "efficiency", "score"], None),
            "can_run": get_path(report, ["resource", "decision", "can_run_requested_profile"], None),
            "recommended_profile": get_path(report, ["resource", "decision", "recommended_profile"], None),
        },
        "coherence_delirium_gate": get_path(report, ["plan", "coherence_delirium_gate"], {}),
        "legacy_port_runtime_enforcement": get_path(report, ["plan", "legacy_port_runtime_enforcement"], {}),
        "review_step_count": goal_review_step_count(report),
        "review_step_basis": "selected_arms_permission_resource_outcome_teacher",
        "maintenance_mode": maintenance_mode_from_report(report),
        "maintenance_mode_basis": "report_goal_or_object_only_default",
        "human_edit_minutes": None,
        "human_edit_minutes_measured": False,
    }


def goal_review_step_count(report: dict[str, Any]) -> int:
    selected = report.get("selected_arms")
    steps = len(selected) if isinstance(selected, list) else 0
    if report.get("permission_envelopes"):
        steps += 1
    if report.get("personality_context"):
        steps += 1
    if report.get("outcome"):
        steps += 1
    if report.get("resource"):
        steps += 1
    if report.get("teacher_needed") is not None:
        steps += 1
    return max(1, steps)


def maintenance_mode_from_report(report: dict[str, Any]) -> str:
    for key in ["maintenance_mode", "maintenance_policy", "maintenance_label"]:
        normalized = normalize_maintenance_mode(report.get(key))
        if normalized:
            return normalized
    text = " ".join(
        str(part or "")
        for part in [
            report.get("goal"),
            report.get("profile"),
            report.get("risk"),
            " ".join(str(arm) for arm in report.get("selected_arms") or []),
        ]
    ).lower()
    if "circle" in text and "seed" in text and "rebuild" in text:
        return "circle_seed_rule_rebuild"
    return "object_only"


def normalize_maintenance_mode(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ordinary": "object_only",
        "ordinary_current": "object_only",
        "baseline": "object_only",
        "object": "object_only",
        "object_only": "object_only",
        "circle": "circle_seed_rule_rebuild",
        "circle_seed_rule": "circle_seed_rule_rebuild",
        "circle_seed_rule_rebuild": "circle_seed_rule_rebuild",
        "seed_rule_rebuild": "circle_seed_rule_rebuild",
    }
    return aliases.get(text, "")


def dedupe_commands(commands: list[list[str]]) -> list[list[str]]:
    seen = set()
    out = []
    for command in commands:
        key = "\0".join(command)
        if key in seen:
            continue
        seen.add(key)
        out.append(command)
    return out


def compact_personality_context(context: dict[str, Any]) -> dict[str, Any]:
    ctx = context.get("context") if isinstance(context.get("context"), dict) else {}
    return {
        "status": context.get("status"),
        "source_report": context.get("source_report"),
        "summary": context.get("summary", {}),
        "compact_core": ctx.get("compact_core", ""),
        "hard_safety_invariants": get_path(ctx, ["reality_contract", "hard_safety_invariants"], [])[:5],
        "anti_drift_rules": (ctx.get("anti_drift_rules") or [])[:5],
    }


def compact_coherence_gate(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": gate.get("trigger_state"),
        "source_trigger_state": gate.get("source_trigger_state"),
        "coherence_score": gate.get("coherence_score"),
        "delirium_score": gate.get("delirium_score"),
        "allows_long_autonomy": gate.get("allows_long_autonomy"),
        "allows_candidate_promotion": gate.get("allows_candidate_promotion"),
        "allows_self_edit": gate.get("allows_self_edit"),
        "allows_capability_expansion": gate.get("allows_capability_expansion"),
        "blockers": gate.get("blockers", []),
        "candidate_blockers": gate.get("candidate_blockers", []),
    }


def compact_runtime_enforcement(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": get_path(report, ["summary", "trigger_state"], None),
        "ready_for_bounded_autonomy": report.get("ready_for_bounded_autonomy"),
        "ready_for_long_autonomy": report.get("ready_for_long_autonomy"),
        "ready_for_candidate_promotion": report.get("ready_for_candidate_promotion"),
        "ready_for_self_evolution": report.get("ready_for_self_evolution"),
        "blockers": report.get("blockers", []),
        "effect_records": get_path(report, ["summary", "effect_records"], None),
        "planforge_nodes": get_path(report, ["summary", "planforge_nodes"], None),
    }


def goal_id(goal: str) -> str:
    return "goal_" + hashlib.sha256(f"{goal}:{time.time()}".encode("utf-8")).hexdigest()[:16]


def get_path(value: Any, path: list[str], default: Any) -> Any:
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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
