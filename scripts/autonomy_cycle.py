"""One SparkStream autonomy cycle.

The cycle observes current RMI ledgers, refreshes benchmark/data inventory,
runs the selected local ratchet profile when requested, checkpoints the result,
and optionally escalates a compact evidence bundle to the sparse teacher.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from autonomy_cycle_runtime import (
    append_jsonl,
    consume_frontier_rotation_request,
    consume_watchdog_override,
    decide_next_action,
    get_path,
    now,
    observe,
    profile_train_limit,
    read_json,
    run_step,
    skipped_step,
    timeout_for_profile,
    update_status,
    write_json,
)
from autonomy_cycle_source_steps import append_source_and_training_inventory_steps
from autonomy_cycle_support import (
    build_self_improvement_queue,
    compact_ledger_entry,
    compact_observation,
    should_call_teacher,
    teacher_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "autonomy_policy.json"
DEFAULT_TEACHER = ROOT / "configs" / "teacher_policy.json"
STATUS_PATH = ROOT / "reports" / "sparkstream_status.json"
LEDGER_PATH = ROOT / "reports" / "autonomy_ledger.jsonl"
QUEUE_PATH = ROOT / "reports" / "self_improvement_queue.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--teacher-policy", default=str(DEFAULT_TEACHER.relative_to(ROOT)))
    parser.add_argument("--profile", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--allow-network-fetch", action="store_true")
    parser.add_argument("--forbid-teacher", action="store_true")
    parser.add_argument("--forbid-network-fetch", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Disable teacher escalation and network fetches for this cycle.")
    parser.add_argument("--out", default="reports/autonomy_cycle_last.json")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    requested_profile = args.profile or policy.get("default_profile", "inner_loop")
    profile = requested_profile
    cycle_id = f"cycle_{int(time.time() * 1000)}"
    update_status(cycle_id, "observing", profile, "Reading current ratchet reports.")
    before = observe()
    decision = decide_next_action(policy, before, requested_profile)
    profile = str(decision.get("profile") or requested_profile)
    teacher_forbidden = bool(args.offline or args.forbid_teacher)
    network_forbidden = bool(args.offline or args.forbid_network_fetch)
    teacher_allowed = False if teacher_forbidden else bool(args.allow_teacher or policy.get("allow_teacher_by_default", False))
    network_allowed = False if network_forbidden else bool(
        args.allow_network_fetch
        or policy.get("allow_network_fetch_by_default", False)
        or decision.get("allow_network_fetch", False)
    )
    write_json(
        ROOT / "reports" / "frontier_policy_status.json",
        {
            "policy": "sparkstream_frontier_policy_status_v0",
            "updated_utc": now(),
            "requested_profile": requested_profile,
            "selected_profile": profile,
            "decision_reason": decision.get("reason"),
            "frontier_pressure": decision.get("frontier_pressure"),
            "frontier_seed": decision.get("frontier_seed"),
            "frontier_family": decision.get("frontier_family"),
            "frontier_eval": decision.get("frontier_eval"),
            "frontier_report": decision.get("frontier_report"),
            "rl_frontier_env": decision.get("rl_frontier_env"),
            "rl_frontier_seed": decision.get("rl_frontier_seed"),
            "pressure_card_id": decision.get("pressure_card_id"),
            "rotation_request_id": decision.get("rotation_request_id"),
            "allow_network_fetch": network_allowed,
            "allow_teacher": teacher_allowed,
            "offline": bool(args.offline),
            "teacher_forbidden": teacher_forbidden,
            "network_fetch_forbidden": network_forbidden,
            "benchmark_discovery_queries": decision.get("benchmark_discovery_queries", []),
        },
    )

    commands: list[dict[str, Any]] = []
    ok = True

    update_status(
        cycle_id,
        "external_inference_audit",
        profile,
        "Verifying that outside intelligence is teacher-only.",
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/external_inference_audit.py",
                "--out",
                "reports/external_inference_audit.json",
            ],
            timeout=120,
            execute=True,
            name="external_inference_audit",
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/license_manager.py",
                "status",
                "--out",
                "reports/license_status.json",
            ],
            timeout=60,
            execute=True,
            name="license_status",
        )
    )

    append_source_and_training_inventory_steps(
        commands=commands,
        policy=policy,
        decision=decision,
        network_allowed=network_allowed,
        policy_path=args.policy,
        cycle_id=cycle_id,
        profile=profile,
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/training_data_inventory.py",
                "--out",
                "reports/training_data_inventory.json",
            ],
            timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
            execute=True,
            name="training_data_inventory",
        )
    )
    cell_lifecycle = policy.get("cell_lifecycle") or {}
    if cell_lifecycle.get("enabled", True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/cell_lifecycle.py",
                    "--policy",
                    str(cell_lifecycle.get("policy", "configs/cell_lifecycle_policy.json")),
                    "--out",
                    str(cell_lifecycle.get("report", "reports/cell_lifecycle.json")),
                    "--markdown-out",
                    str(cell_lifecycle.get("markdown_report", "reports/cell_lifecycle.md")),
                    "--prune-plan-out",
                    str(cell_lifecycle.get("prune_plan_report", "reports/cell_lifecycle_prune_plan.json")),
                ],
                timeout=120,
                execute=True,
                name="cell_lifecycle",
            )
        )
    personality_core = policy.get("personality_core") or {}
    if personality_core.get("enabled", True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/personality_core.py",
                    "--policy",
                    str(personality_core.get("policy", "configs/personality_core_policy.json")),
                    "--out",
                    str(personality_core.get("report", "reports/personality_core.json")),
                    "--markdown-out",
                    str(personality_core.get("markdown_report", "reports/personality_core.md")),
                    "--manifest-out",
                    str(personality_core.get("manifest_report", "reports/personality_core_training_manifest.jsonl")),
                ],
                timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
                execute=True,
                name="personality_core",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/personality_context_builder.py",
                    "--policy",
                    str(personality_core.get("policy", "configs/personality_core_policy.json")),
                    "--task",
                    "autonomy_cycle",
                    "--prompt",
                    "Launch orientation, autonomy guardrails, self-evolution drift prevention, and belief update governance.",
                    "--out",
                    str(get_path(personality_core, ["context_report"], "reports/personality_context_last.json")),
                ],
                timeout=120,
                execute=True,
                name="personality_context_builder",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/personality_drift_eval.py",
                    "--policy",
                    str(get_path(personality_core, ["drift_eval_policy"], "configs/personality_drift_eval.json")),
                    "--out",
                    str(get_path(personality_core, ["drift_eval_report"], "reports/personality_drift_eval.json")),
                ],
                timeout=300,
                execute=True,
                name="personality_drift_eval",
                allow_failure=False,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/multi_turn_conversation_benchmark.py",
                    "--config",
                    str(get_path(personality_core, ["conversation_benchmark_policy"], "configs/multi_turn_conversation_benchmark.json")),
                    "--out",
                    str(get_path(personality_core, ["conversation_benchmark_report"], "reports/multi_turn_conversation_benchmark.json")),
                    "--markdown-out",
                    str(
                        get_path(
                            personality_core,
                            ["conversation_benchmark_markdown_report"],
                            "reports/multi_turn_conversation_benchmark.md",
                        )
                    ),
                ],
                timeout=420,
                execute=True,
                name="multi_turn_conversation_benchmark",
                allow_failure=False,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/belief_update_governor.py",
                    "--policy",
                    str(get_path(personality_core, ["belief_update_policy"], "configs/belief_update_policy.json")),
                    "--status-only",
                    "--out",
                    str(get_path(personality_core, ["belief_update_report"], "reports/belief_update_governance.json")),
                ],
                timeout=120,
                execute=True,
                name="belief_update_governance",
                allow_failure=False,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/personality_runtime_audit.py",
                    "--policy",
                    str(personality_core.get("policy", "configs/personality_core_policy.json")),
                    "--out",
                    str(get_path(personality_core, ["runtime_audit_report"], "reports/personality_runtime_audit.json")),
                ],
                timeout=180,
                execute=True,
                name="personality_runtime_audit",
                allow_failure=False,
            )
        )
    if get_path(policy, ["local_rom_assets", "enabled"], True):
        rom_stage_command = [
            sys.executable,
            "scripts/stage_local_rom_assets.py",
            "--policy",
            str(get_path(policy, ["local_rom_assets", "policy"], "configs/local_rom_policy.json")),
            "--dest",
            str(get_path(policy, ["local_rom_assets", "destination"], "data/local_roms")),
            "--out",
            str(get_path(policy, ["local_rom_assets", "report"], "reports/local_rom_staging_report.json")),
            "--inventory-out",
            str(get_path(policy, ["local_rom_assets", "inventory_report"], "reports/game_asset_inventory.json")),
        ]
        for source_root in get_path(policy, ["local_rom_assets", "source_roots"], []):
            rom_stage_command.extend(["--source-root", str(source_root)])
        if get_path(policy, ["local_rom_assets", "stage_on_cycle"], True):
            rom_stage_command.append("--execute")
        commands.append(
            run_step(
                rom_stage_command,
                timeout=600,
                execute=True,
                name="local_rom_asset_staging",
                allow_failure=True,
            )
        )
        if get_path(policy, ["hive", "refresh_public_contribution_each_cycle"], True):
            commands.append(
                run_step(
                    [
                        sys.executable,
                        "scripts/public_hive_contributor.py",
                        "status",
                        "--out",
                        str(get_path(policy, ["hive", "public_contribution_status"], "reports/public_hive_contribution_status.json")),
                    ],
                    timeout=60,
                    execute=True,
                    name="public_hive_contribution_status",
                    allow_failure=True,
                )
            )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/local_rom_registry.py",
                "--out",
                "reports/local_rom_registry.json",
            ],
            timeout=300,
            execute=True,
            name="local_rom_registry",
            allow_failure=True,
        )
    )
    if get_path(policy, ["synthetic_data", "enabled"], True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/synthetic_data_curator.py",
                    "--policy",
                    str(get_path(policy, ["synthetic_data", "policy"], "configs/synthetic_data_policy.json")),
                    "--blend-total",
                    str(profile_train_limit(profile)),
                    "--out",
                    "reports/synthetic_data_curator.json",
                ],
                timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
                execute=True,
                name="synthetic_data_curator",
                allow_failure=True,
            )
        )
    rl_command = [
        sys.executable,
        "scripts/rl_benchmark_registry.py",
        "--refresh-local",
        "--out",
        "reports/rl_benchmark_registry.json",
    ]
    rl_queries = get_path(policy, ["rl_benchmarks", "autonomous_discovery_queries"], [])
    commands.append(
        run_step(
            rl_command,
            timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
            execute=True,
            name="rl_benchmark_registry",
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/legacy_rl_environment_admission.py",
                "--out",
                "reports/legacy_rl_environment_admission.json",
                "--markdown-out",
                "reports/legacy_rl_environment_admission.md",
            ],
            timeout=180,
            execute=True,
            name="legacy_rl_environment_admission",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/legacy_rl_smoke_plan.py",
                "--admission",
                "reports/legacy_rl_environment_admission.json",
                "--out",
                "reports/legacy_rl_smoke_plan.json",
                "--plan-out",
                "data/rl_smoke/legacy_rl_smoke_plan.jsonl",
                "--limit",
                "16",
            ],
            timeout=120,
            execute=True,
            name="legacy_rl_smoke_plan",
            allow_failure=True,
        )
    )
    if get_path(policy, ["minecraft_rl", "enabled"], True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/minecraft_runtime_probe.py",
                    "--policy",
                    str(get_path(policy, ["minecraft_rl", "policy"], "configs/minecraft_rl_policy.json")),
                    "--out",
                    str(get_path(policy, ["minecraft_rl", "report"], "reports/minecraft_runtime_probe.json")),
                ],
                timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
                execute=True,
                name="minecraft_runtime_probe",
                allow_failure=True,
            )
        )
    if network_allowed and rl_queries:
        queries_per_cycle = int(get_path(policy, ["rl_benchmarks", "queries_per_cycle"], 1))
        for index, query in enumerate(rl_queries[: max(1, queries_per_cycle)]):
            rl_discovery_command = [
                sys.executable,
                "scripts/rl_benchmark_registry.py",
                "--allow-network-discovery",
                "--discover-query",
                str(query),
                "--discover-limit",
                str(get_path(policy, ["rl_benchmarks", "discover_limit"], 10)),
                "--out",
                "reports/rl_benchmark_registry.json",
            ]
            if get_path(policy, ["rl_benchmarks", "autonomous_import_approved_sources"], False):
                rl_discovery_command.extend(
                    [
                        "--import-approved",
                        "--allow-network-import",
                        "--max-imports",
                        str(get_path(policy, ["rl_benchmarks", "max_imports_per_cycle"], 2)),
                    ]
                )
            commands.append(
                run_step(
                    rl_discovery_command,
                    timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
                    execute=True,
                    name=f"rl_benchmark_discovery_{index + 1}",
                    allow_failure=True,
                )
            )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/resource_governor.py",
                "--profile",
                profile,
                "--out",
                "reports/resource_governor.json",
            ],
            timeout=120,
            execute=True,
            name="resource_governor",
        )
    )
    if get_path(policy, ["hive", "enabled"], True) and get_path(policy, ["hive", "refresh_each_cycle"], True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/hive_node.py",
                    "--policy",
                    str(get_path(policy, ["hive", "policy"], "configs/hive_policy.json")),
                    "probe",
                    "--out",
                    str(get_path(policy, ["hive", "status"], "reports/hive_status.json")),
                    "--peers-out",
                    str(get_path(policy, ["hive", "peers"], "reports/hive_peers.json")),
                ],
                timeout=120,
                execute=True,
                name="hive_node_probe",
                allow_failure=True,
            )
        )
        if get_path(policy, ["hive", "schedule_each_cycle"], True):
            hive_scheduler_command = [
                sys.executable,
                "scripts/hive_scheduler.py",
                "--policy",
                str(get_path(policy, ["hive", "policy"], "configs/hive_policy.json")),
                "--out",
                str(get_path(policy, ["hive", "scheduler"], "reports/hive_scheduler.json")),
            ]
            if args.execute and get_path(policy, ["hive", "execute_remote_tasks_by_default"], False):
                hive_scheduler_command.append("--execute")
            if get_path(policy, ["hive", "plan_worker_chunks_each_cycle"], True):
                hive_scheduler_command.append("--worker-chunks")
            if args.execute and get_path(policy, ["hive", "execute_worker_chunks_by_default"], False):
                if "--execute" not in hive_scheduler_command:
                    hive_scheduler_command.append("--execute")
                if "--worker-chunks" not in hive_scheduler_command:
                    hive_scheduler_command.append("--worker-chunks")
            commands.append(
                run_step(
                    hive_scheduler_command,
                    timeout=120,
                    execute=True,
                    name="hive_scheduler",
                    allow_failure=True,
                )
            )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/hive_operator_os.py",
                    "--config",
                    "configs/hive_operator_os.json",
                    "--db",
                    "reports/hive_work_board.sqlite",
                    "--out",
                    "reports/hive_operator_os.json",
                    "--markdown-out",
                    "reports/hive_operator_os.md",
                ],
                timeout=120,
                execute=True,
                name="hive_operator_os",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/hive_work_board_executor.py",
                    "--status",
                    "--out",
                    "reports/hive_work_board_executor.json",
                    "--markdown-out",
                    "reports/hive_work_board_executor.md",
                ],
                timeout=120,
                execute=True,
                name="hive_work_board_executor_status",
                allow_failure=True,
            )
        )
    if get_path(policy, ["performance_optimizer", "enabled"], True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/runtime_bottleneck_optimizer_worker_chunk_plan.py",
                    "--out",
                    "reports/runtime_bottleneck_optimizer_worker_chunk_plan.json",
                    "--markdown-out",
                    "reports/runtime_bottleneck_optimizer_worker_chunk_plan.md",
                    "--lease-out",
                    "reports/runtime_bottleneck_optimizer_worker_chunk_leases.jsonl",
                ],
                timeout=120,
                execute=bool(get_path(policy, ["performance_optimizer", "refresh_each_cycle"], True)),
                name="runtime_bottleneck_optimizer_worker_chunk_plan",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/performance_optimizer.py",
                    "--policy",
                    str(get_path(policy, ["performance_optimizer", "policy"], "configs/performance_policy.json")),
                    "--out",
                    str(get_path(policy, ["performance_optimizer", "report"], "reports/performance_optimizer.json")),
                    "--markdown-out",
                    str(get_path(policy, ["performance_optimizer", "markdown_report"], "reports/performance_optimizer.md")),
                ],
                timeout=120,
                execute=bool(get_path(policy, ["performance_optimizer", "refresh_each_cycle"], True)),
                name="performance_optimizer",
                allow_failure=True,
            )
        )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/octopus_router.py",
                "--router-head-report",
                "reports/octopus_router_head_report.json",
                "--router-head-eval",
                "reports/octopus_router_head_eval.json",
                "--out",
                "reports/octopus_router_report.json",
            ],
            timeout=120,
            execute=True,
            name="octopus_router_refresh",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/autonomous_goal_runner.py",
                "--goal",
                str(decision.get("goal") or "Improve the current frontier efficiently using octopus router arms and preserve regressions."),
                "--profile",
                profile,
                "--out",
                "reports/autonomous_goal_last.json",
            ],
            timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
            execute=True,
            name="autonomous_goal_route_plan",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/arm_lifecycle_manager.py",
                "--out",
                "reports/arm_lifecycle_governance.json",
            ],
            timeout=120,
            execute=True,
            name="arm_lifecycle_governance",
            allow_failure=True,
        )
    )
    if get_path(policy, ["arm_suckers", "enabled"], True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/arm_sucker_registry.py",
                    "--policy",
                    str(get_path(policy, ["arm_suckers", "policy"], "configs/arm_sucker_policy.json")),
                    "--out",
                    str(get_path(policy, ["arm_suckers", "report"], "reports/arm_sucker_registry.json")),
                    "--markdown-out",
                    str(get_path(policy, ["arm_suckers", "markdown_report"], "reports/arm_sucker_registry.md")),
                ],
                timeout=120,
                execute=True,
                name="arm_sucker_registry",
                allow_failure=True,
            )
        )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/attd_dirty_workspace_checkpoint.py",
                "--policy",
                args.policy,
                "--attd-policy",
                str(get_path(policy, ["attd", "policy"], "configs/attd_policy.json")),
                "--attd-report",
                str(get_path(policy, ["attd", "report"], "reports/attd_report.json")),
                "--packets-out",
                str(get_path(policy, ["attd", "maintenance_packets"], "reports/attd_maintenance_packets.json")),
                "--markdown-out",
                str(get_path(policy, ["attd", "markdown_report"], "reports/attd_report.md")),
                "--out",
                str(get_path(policy, ["attd", "dirty_checkpoint_report"], "reports/attd_dirty_workspace_checkpoint.json")),
            ],
            timeout=360,
            execute=True,
            name="attd_dirty_workspace_checkpoint",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/attd_analyzer.py",
                "--policy",
                str(get_path(policy, ["attd", "policy"], "configs/attd_policy.json")),
                "--out",
                str(get_path(policy, ["attd", "report"], "reports/attd_report.json")),
                "--packets-out",
                str(get_path(policy, ["attd", "maintenance_packets"], "reports/attd_maintenance_packets.json")),
                "--markdown-out",
                str(get_path(policy, ["attd", "markdown_report"], "reports/attd_report.md")),
            ],
            timeout=120,
            execute=True,
            name="attd_analyzer",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/legacy_port_runtime_enforcer.py",
                "--out",
                "reports/legacy_port_runtime_enforcement.json",
            ],
            timeout=180,
            execute=True,
            name="legacy_port_runtime_enforcement_post_attd",
            allow_failure=True,
        )
    )
    resource_report = read_json(ROOT / "reports" / "resource_governor.json")
    if decision["run_profile"] and not get_path(resource_report, ["decision", "can_run_requested_profile"], True):
        decision["run_profile"] = False
        decision["reason"] = "resource_governor_throttle"
        decision["resource_throttle_reasons"] = get_path(resource_report, ["decision", "throttle_reasons"], [])
    arm_governance = read_json(ROOT / "reports" / "arm_lifecycle_governance.json")
    if decision["run_profile"] and not arm_governance.get("ready_for_long_autonomy", False):
        decision["run_profile"] = False
        decision["reason"] = "arm_lifecycle_governance_blocked"
        decision["teacher_reason"] = "safety_or_governance_uncertainty"
        decision["arm_lifecycle_blockers"] = get_path(arm_governance, ["validation", "issues"], [])
    attd_report = read_json(ROOT / "reports" / "attd_report.json")
    if decision["run_profile"] and not get_path(attd_report, ["governance", "allows_long_autonomy"], False):
        decision["run_profile"] = False
        decision["reason"] = "attd_governance_blocked"
        decision["teacher_reason"] = "safety_or_governance_uncertainty"
        decision["attd_trigger_state"] = attd_report.get("trigger_state")
        decision["attd_score"] = attd_report.get("attd_score")
        decision["attd_maintenance_packets"] = get_path(attd_report, ["maintenance_packets_path"], "reports/attd_maintenance_packets.json")
    coherence_gate = read_json(ROOT / "reports" / "coherence_delirium_gate.json")
    if decision["run_profile"] and not coherence_gate.get("allows_long_autonomy", False):
        decision["run_profile"] = False
        decision["reason"] = "coherence_delirium_governance_blocked"
        decision["teacher_reason"] = "safety_or_governance_uncertainty"
        decision["coherence_trigger_state"] = coherence_gate.get("trigger_state")
        decision["coherence_score"] = coherence_gate.get("coherence_score")
        decision["delirium_score"] = coherence_gate.get("delirium_score")
        decision["coherence_blockers"] = coherence_gate.get("blockers", [])
    runtime_enforcement = read_json(ROOT / "reports" / "legacy_port_runtime_enforcement.json")
    if decision["run_profile"] and not runtime_enforcement.get("ready_for_long_autonomy", False):
        decision["run_profile"] = False
        decision["reason"] = "legacy_port_runtime_enforcement_blocked"
        decision["teacher_reason"] = "safety_or_governance_uncertainty"
        decision["legacy_runtime_enforcement_state"] = get_path(runtime_enforcement, ["summary", "trigger_state"], None)
        decision["legacy_runtime_enforcement_blockers"] = runtime_enforcement.get("blockers", [])

    if args.execute and decision["run_profile"]:
        update_status(cycle_id, "running_profile", profile, f"Running {profile} ratchet profile.")
        profile_command = [
            sys.executable,
            "scripts/run_training_ratchet_profile.py",
            "--profile",
            profile,
            "--out",
            "reports/training_ratchet_profile_run.json",
        ]
        if decision.get("frontier_seed") is not None:
            profile_command.extend(["--frontier-seed", str(decision["frontier_seed"])])
        if decision.get("frontier_family"):
            profile_command.extend(["--frontier-family", str(decision["frontier_family"])])
        if decision.get("frontier_eval"):
            profile_command.extend(["--frontier-eval", str(decision["frontier_eval"])])
        if decision.get("frontier_report"):
            profile_command.extend(["--frontier-report", str(decision["frontier_report"])])
        if decision.get("rl_frontier_env"):
            profile_command.extend(["--rl-frontier-env", str(decision["rl_frontier_env"])])
        if decision.get("rl_frontier_seed") is not None:
            profile_command.extend(["--rl-frontier-seed", str(decision["rl_frontier_seed"])])
        if decision.get("pressure_card_id"):
            profile_command.extend(["--pressure-card-id", str(decision["pressure_card_id"])])
        if decision.get("force_frontier_generation"):
            profile_command.append("--force-frontier-generation")
        if teacher_allowed:
            profile_command.append("--allow-teacher")
        commands.append(
            run_step(
                profile_command,
                timeout=timeout_for_profile(policy, profile),
                execute=True,
                name=f"profile_{profile}",
            )
        )
    elif decision["run_profile"]:
        planned_profile_command = [
            sys.executable,
            "scripts/run_training_ratchet_profile.py",
            "--profile",
            profile,
        ]
        if decision.get("frontier_seed") is not None:
            planned_profile_command.extend(["--frontier-seed", str(decision["frontier_seed"])])
        if decision.get("frontier_family"):
            planned_profile_command.extend(["--frontier-family", str(decision["frontier_family"])])
        if decision.get("rl_frontier_env"):
            planned_profile_command.extend(["--rl-frontier-env", str(decision["rl_frontier_env"])])
        if decision.get("rl_frontier_seed") is not None:
            planned_profile_command.extend(["--rl-frontier-seed", str(decision["rl_frontier_seed"])])
        if decision.get("pressure_card_id"):
            planned_profile_command.extend(["--pressure-card-id", str(decision["pressure_card_id"])])
        if decision.get("force_frontier_generation"):
            planned_profile_command.append("--force-frontier-generation")
        if teacher_allowed:
            planned_profile_command.append("--allow-teacher")
        commands.append(
            skipped_step(
                f"profile_{profile}",
                "planned_but_not_executed_without_execute_flag",
                planned_profile_command,
            )
        )

    if args.execute:
        update_status(cycle_id, "refreshing_ratchet", profile, "Refreshing compiled capability ratchet.")
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/run_capability_ratchet.py",
                    "--mutated-babylm-report",
                    str(decision.get("frontier_report") or "reports/babylm_mutated_holdout_seed55_stateful_grammar_state_frontier.json"),
                    "--mutated-babylm-eval",
                    str(decision.get("frontier_eval") or "data/babylm_mutated_holdout_seed55.jsonl"),
                    "--skip-holdout-generation",
                    "--out",
                    "reports/capability_ratchet_run.json",
                ],
                timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
                execute=True,
                name="capability_ratchet_refresh",
            )
        )

    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/candidate_promotion_gate.py",
                "--out",
                "reports/candidate_promotion_gate.json",
            ],
            timeout=120,
            execute=True,
            name="candidate_promotion_gate",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/promotion_closure.py",
                "--out",
                "reports/promotion_closure.json",
            ],
            timeout=120,
            execute=True,
            name="promotion_closure",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/ai_grand_prix_spec_digest.py",
                "--out",
                "reports/ai_grand_prix_spec_digest.json",
            ],
            timeout=120,
            execute=True,
            name="ai_grand_prix_spec_digest",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/python_runtime_compatibility.py",
                "--out",
                "reports/python_runtime_compatibility.json",
            ],
            timeout=60,
            execute=True,
            name="python_runtime_compatibility",
            allow_failure=True,
        )
    )
    bottleneck_command = [
        sys.executable,
        "scripts/candidate_bottleneck_reducer.py",
        "--out",
        "reports/candidate_bottleneck_reducer.json",
    ]
    if args.execute:
        bottleneck_command.append("--fix")
    commands.append(
        run_step(
            bottleneck_command,
            timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
            execute=True,
            name="candidate_bottleneck_reducer",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/autonomy_launch_readiness.py",
                "--profile",
                profile,
                "--out",
                "reports/autonomy_launch_readiness.json",
            ],
            timeout=120,
            execute=True,
            name="autonomy_launch_readiness",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/capability_matrix.py",
                "--out",
                "reports/capability_matrix.json",
            ],
            timeout=120,
            execute=True,
            name="capability_matrix_refresh",
            allow_failure=True,
        )
    )
    if get_path(policy, ["synthetic_benchmarks", "enabled"], True):
        synthetic_command = [
            sys.executable,
            "scripts/synthetic_benchmark_factory.py",
            "--policy",
            str(get_path(policy, ["synthetic_benchmarks", "policy"], "configs/synthetic_benchmark_policy.json")),
            "--out",
            str(get_path(policy, ["synthetic_benchmarks", "report"], "reports/synthetic_benchmark_factory.json")),
            "--markdown-out",
            str(get_path(policy, ["synthetic_benchmarks", "markdown_report"], "reports/synthetic_benchmark_factory.md")),
        ]
        if get_path(policy, ["synthetic_benchmarks", "write_cards"], True):
            synthetic_command.append("--write-cards")
        commands.append(
            run_step(
                synthetic_command,
                timeout=120,
                execute=True,
                name="synthetic_benchmark_factory",
                allow_failure=True,
            )
        )
    if get_path(policy, ["multi_stream", "enabled"], True):
        multi_stream_command = [
            sys.executable,
            "scripts/multi_stream_trace_factory.py",
            "--policy",
            str(get_path(policy, ["multi_stream", "policy"], "configs/multi_stream_policy.json")),
            "--out",
            str(get_path(policy, ["multi_stream", "report"], "reports/multi_stream_trace_factory.json")),
            "--markdown-out",
            str(get_path(policy, ["multi_stream", "markdown_report"], "reports/multi_stream_trace_factory.md")),
        ]
        if get_path(policy, ["multi_stream", "write_cards"], True):
            multi_stream_command.append("--write-cards")
        commands.append(
            run_step(
                multi_stream_command,
                timeout=120,
                execute=True,
                name="multi_stream_trace_factory",
                allow_failure=True,
            )
        )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/benchmaxx_curriculum.py",
                "--config",
                str(get_path(policy, ["benchmaxx_curriculum", "config"], "configs/benchmaxx_curriculum.json")),
                "--out",
                str(get_path(policy, ["benchmaxx_curriculum", "report"], "reports/benchmaxx_curriculum.json")),
                "--markdown-out",
                str(get_path(policy, ["benchmaxx_curriculum", "markdown_report"], "reports/benchmaxx_curriculum.md")),
            ],
            timeout=120,
            execute=True,
            name="benchmaxx_curriculum_refresh",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/viea_autonomy_spine.py",
                "--max-steps",
                "64",
                "--timeout-seconds",
                str(int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800))),
                "--out",
                "reports/viea_autonomy_spine.json",
                "--markdown-out",
                "reports/viea_autonomy_spine.md",
            ],
            timeout=max(1800, int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800))),
            execute=True,
            name="viea_autonomy_spine",
            allow_failure=True,
        )
    )
    viea_action_command = [
        sys.executable,
        "scripts/viea_action_executor.py",
        "--max-actions",
        "1",
        "--max-steps",
        "1",
        "--timeout-seconds",
        str(max(21600, int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)))),
        "--resume",
        "--out",
        "reports/viea_action_executor.json",
        "--markdown-out",
        "reports/viea_action_executor.md",
    ]
    if args.execute:
        viea_action_command.append("--execute")
    if teacher_allowed:
        viea_action_command.append("--allow-teacher")
    commands.append(
        run_step(
            viea_action_command,
            timeout=max(21600, int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800))),
            execute=True,
            name="viea_action_executor",
            allow_failure=True,
        )
    )
    if get_path(policy, ["benchmark_adapter_factory", "enabled"], True):
        attd_allows_adapter_writes = get_path(
            read_json(ROOT / str(get_path(policy, ["attd", "report"], "reports/attd_report.json"))),
            ["governance", "allows_adapter_card_writes"],
            False,
        )
        adapter_command = [
            sys.executable,
            "scripts/benchmark_adapter_factory.py",
            "--config",
            str(get_path(policy, ["benchmark_adapter_factory", "config"], "configs/benchmark_adapter_factory.json")),
            "--out",
            str(get_path(policy, ["benchmark_adapter_factory", "report"], "reports/benchmark_adapter_factory.json")),
            "--markdown-out",
            str(
                get_path(
                    policy,
                    ["benchmark_adapter_factory", "markdown_report"],
                    "reports/benchmark_adapter_factory.md",
                )
            ),
        ]
        if get_path(policy, ["benchmark_adapter_factory", "write_cards"], True) and attd_allows_adapter_writes:
            adapter_command.append("--write-cards")
        commands.append(
            run_step(
                adapter_command,
                timeout=120,
                execute=True,
                name="benchmark_adapter_factory",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/benchmark_pantry_unblocker.py",
                    "--factory",
                    str(get_path(policy, ["benchmark_adapter_factory", "report"], "reports/benchmark_adapter_factory.json")),
                    "--out",
                    "reports/benchmark_pantry_unblocker.json",
                ],
                timeout=120,
                execute=True,
                name="benchmark_pantry_unblocker",
                allow_failure=True,
            )
        )
    if get_path(policy, ["architecture_search", "enabled"], True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/architecture_experiment_governor.py",
                    "--space",
                    str(get_path(policy, ["architecture_search", "space"], "configs/architecture_search_space.json")),
                    "--out",
                    str(get_path(policy, ["architecture_search", "report"], "reports/architecture_experiment_governance.json")),
                    "--markdown-out",
                    str(
                        get_path(
                            policy,
                            ["architecture_search", "markdown_report"],
                            "reports/architecture_experiment_governance.md",
                        )
                    ),
                ],
                timeout=120,
                execute=True,
                name="architecture_experiment_governance",
                allow_failure=True,
            )
        )
        experiment_command = [
            sys.executable,
            "scripts/architecture_experiment_runner.py",
            "--governance",
            str(get_path(policy, ["architecture_search", "report"], "reports/architecture_experiment_governance.json")),
            "--max-experiments",
            str(get_path(policy, ["architecture_search", "runner_max_experiments"], 1)),
            "--max-commands",
            str(get_path(policy, ["architecture_search", "runner_max_commands"], 1)),
            "--timeout-seconds",
            str(get_path(policy, ["command_timeouts_seconds", "smoke"], 600)),
            "--out",
            str(get_path(policy, ["architecture_search", "runner_report"], "reports/architecture_experiment_runner.json")),
        ]
        if args.execute and get_path(policy, ["architecture_search", "execute_runner_when_allowed"], True):
            experiment_command.append("--execute")
        commands.append(
            run_step(
                experiment_command,
                timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
                execute=True,
                name="architecture_experiment_runner",
                allow_failure=True,
            )
        )
    if get_path(policy, ["autoresearch_gap_audit", "enabled"], True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/autoresearch_gap_audit.py",
                    "--policy",
                    str(get_path(policy, ["autoresearch_gap_audit", "policy"], "configs/autoresearch_loop_policy.json")),
                    "--out",
                    str(get_path(policy, ["autoresearch_gap_audit", "report"], "reports/autoresearch_gap_audit.json")),
                    "--markdown-out",
                    str(
                        get_path(
                            policy,
                            ["autoresearch_gap_audit", "markdown_report"],
                            "reports/autoresearch_gap_audit.md",
                        )
                    ),
                ],
                timeout=120,
                execute=True,
                name="autoresearch_gap_audit",
                allow_failure=True,
            )
        )
    if get_path(policy, ["loop_closure_harvester", "enabled"], True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/loop_closure_harvester.py",
                    "--minimum-recurrence",
                    str(get_path(policy, ["loop_closure_harvester", "minimum_recurrence"], 3)),
                    "--out",
                    str(get_path(policy, ["loop_closure_harvester", "report"], "reports/loop_closure_harvester.json")),
                    "--markdown-out",
                    str(
                        get_path(
                            policy,
                            ["loop_closure_harvester", "markdown_report"],
                            "reports/loop_closure_harvester.md",
                        )
                    ),
                ],
                timeout=120,
                execute=True,
                name="loop_closure_harvester",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/loop_closure_tool_promoter.py",
                    "--harvester",
                    str(get_path(policy, ["loop_closure_harvester", "report"], "reports/loop_closure_harvester.json")),
                    "--registry",
                    "reports/tool_registry.json",
                    "--out",
                    "reports/loop_closure_tool_promoter.json",
                ],
                timeout=120,
                execute=True,
                name="loop_closure_tool_promoter",
                allow_failure=True,
            )
        )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/native_voice_training_manifest.py",
                "--out",
                "reports/native_voice_training_manifest.json",
                *(["--allow-network-fetch"] if network_allowed else []),
            ],
            timeout=180,
            execute=True,
            name="native_voice_training_manifest",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/native_voice_bootstrap_learner.py",
                "--manifest",
                "reports/native_voice_training_manifest.json",
                "--stt-out",
                "reports/native_stt_decoder.json",
                "--tts-out",
                "reports/native_tts_generator.json",
            ],
            timeout=120,
            execute=True,
            name="native_voice_bootstrap_learner",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/native_voice_io.py",
                "--out",
                "reports/native_voice_io.json",
            ],
            timeout=60,
            execute=True,
            name="native_voice_io",
            allow_failure=True,
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/transfer_eval_suite.py",
                "--out",
                "reports/transfer_eval_suite.json",
            ],
            timeout=120,
            execute=True,
            name="transfer_eval_suite",
            allow_failure=True,
        )
    )
    if get_path(policy, ["arm_transfer", "enabled"], True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/arm_transfer_planner.py",
                    "--out",
                    str(get_path(policy, ["arm_transfer", "report"], "reports/arm_transfer_plan.json")),
                    "--markdown-out",
                    str(get_path(policy, ["arm_transfer", "markdown_report"], "reports/arm_transfer_plan.md")),
                ],
                timeout=120,
                execute=True,
                name="arm_transfer_plan",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/transfer_artifact_builder.py",
                    "--plan",
                    str(get_path(policy, ["arm_transfer", "report"], "reports/arm_transfer_plan.json")),
                    "--artifact-dir",
                    str(get_path(policy, ["arm_transfer", "artifact_dir"], "reports/transfer_artifacts")),
                    "--out",
                    str(get_path(policy, ["arm_transfer", "artifact_report"], "reports/arm_transfer_artifacts.json")),
                ],
                timeout=120,
                execute=True,
                name="arm_transfer_artifacts",
                allow_failure=True,
            )
        )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/model_growth_gate.py",
                "--out",
                "reports/model_growth_gate.json",
            ],
            timeout=120,
            execute=True,
            name="model_growth_gate",
            allow_failure=True,
        )
    )
    if get_path(policy, ["self_evolution", "enabled"], True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/self_evolution_governor.py",
                    "--policy",
                    str(get_path(policy, ["self_evolution", "policy"], "configs/self_evolution_policy.json")),
                    "--out",
                    str(get_path(policy, ["self_evolution", "report"], "reports/self_evolution_governance.json")),
                    "--markdown-out",
                    str(
                        get_path(
                            policy,
                            ["self_evolution", "markdown_report"],
                            "reports/self_evolution_governance.md",
                        )
                    ),
                ],
                timeout=120,
                execute=True,
                name="self_evolution_governance",
                allow_failure=True,
            )
        )
        self_evolution = read_json(ROOT / str(get_path(policy, ["self_evolution", "report"], "reports/self_evolution_governance.json")))
        if (
            args.execute
            and teacher_allowed
            and get_path(policy, ["self_evolution", "auto_apply_when_policy_allows"], True)
            and get_path(self_evolution, ["teacher_apply", "allowed_now"], False)
        ):
            commands.append(
                run_step(
                    [
                        sys.executable,
                        "scripts/teacher_self_edit_runner.py",
                        "--execute",
                        "--allow-teacher",
                        "--reason",
                        str(get_path(self_evolution, ["teacher_apply", "triggered_by", "primary_reason"], None) or get_path(self_evolution, ["teacher_apply", "triggered_by", "wall_type"], None) or "architecture_wall"),
                        "--out",
                        str(
                            get_path(
                                policy,
                                ["self_evolution", "teacher_self_edit_report"],
                                "reports/teacher_self_edit_last.json",
                            )
                        ),
                    ],
                    timeout=int(get_path(policy, ["command_timeouts_seconds", "teacher"], 1800)),
                    execute=True,
                    name="guarded_teacher_self_edit",
                    allow_failure=True,
                )
            )
        else:
            commands.append(
                skipped_step(
                    "guarded_teacher_self_edit",
                    "self_evolution_policy_not_triggered_or_teacher_not_allowed",
                    [
                        sys.executable,
                        "scripts/teacher_self_edit_runner.py",
                        "--execute",
                        "--allow-teacher",
                    ],
                )
            )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/teacher_self_edit_proof.py",
                    "--out",
                    "reports/teacher_self_edit_proof.json",
                ],
                timeout=120,
                execute=True,
                name="teacher_self_edit_proof",
                allow_failure=True,
            )
        )

    after = observe()
    teacher_result = None
    teacher_needed = should_call_teacher(policy, after, decision)
    if teacher_needed:
        evidence = teacher_evidence(after, decision)
        if teacher_allowed:
            update_status(cycle_id, "teacher", profile, "Escalating compact evidence bundle to teacher.")
        else:
            update_status(cycle_id, "teacher_queued", profile, "Teacher escalation queued by policy.")
        teacher_step = run_step(
            [
                sys.executable,
                "scripts/teacher_oracle.py",
                "--reason",
                decision["teacher_reason"],
                "--mode",
                "proposal",
                "--local-evidence",
                *evidence,
                "--out",
                "reports/teacher_oracle_last.json",
                *(["--allow-teacher"] if teacher_allowed else []),
            ],
            timeout=int(get_path(policy, ["command_timeouts_seconds", "teacher"], 1800)),
            execute=True,
            name="teacher_oracle",
            allow_failure=True,
        )
        commands.append(teacher_step)
        teacher_result = read_json(ROOT / "reports" / "teacher_oracle_last.json")

    update_status(cycle_id, "checkpointing", profile, "Creating checkpoint manifest.")
    checkpoint_status = "promoted" if get_path(after, ["candidate_gate", "promote"], False) else "observed"
    checkpoint = run_step(
        [
            sys.executable,
            "scripts/checkpoint_registry.py",
            "create",
            "--label",
            f"sparkstream_{profile}",
            "--reason",
            decision["reason"],
            "--profile",
            profile,
            "--status",
            checkpoint_status,
            "--out",
            "reports/checkpoint_last.json",
        ],
        timeout=120,
        execute=True,
        name="checkpoint_create",
    )
    commands.append(checkpoint)
    if get_path(policy, ["checkpoint_backup", "enabled"], True) and get_path(
        policy, ["checkpoint_backup", "run_after_checkpoint_create"], True
    ):
        backup_command = [
            sys.executable,
            "scripts/checkpoint_backup_manager.py",
            "--policy",
            str(get_path(policy, ["checkpoint_backup", "policy"], "configs/checkpoint_backup_policy.json")),
            "--if-promoted",
            "--provider",
            str(get_path(policy, ["checkpoint_backup", "provider"], "all")),
            "--out",
            str(get_path(policy, ["checkpoint_backup", "report"], "reports/checkpoint_backup_last.json")),
        ]
        if args.execute and get_path(policy, ["checkpoint_backup", "execute_on_accept"], True):
            backup_command.append("--execute")
        commands.append(
            run_step(
                backup_command,
                timeout=120,
                execute=True,
                name="checkpoint_backup_if_accepted",
                allow_failure=True,
            )
        )
    if get_path(after, ["candidate_gate", "promote"], False):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/update_manager.py",
                    "create",
                    "--if-promoted",
                    "--out",
                    "reports/update_offer_current.json",
                ],
                timeout=120,
                execute=True,
                name="update_offer_create",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/update_manager.py",
                    "apply",
                    "--mode",
                    "auto",
                    "--execute",
                    "--out",
                    "reports/update_apply_last.json",
                ],
                timeout=120,
                execute=True,
                name="update_auto_apply",
                allow_failure=True,
            )
        )
    else:
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/update_manager.py",
                    "status",
                    "--out",
                    "reports/update_status.json",
                ],
                timeout=120,
                execute=True,
                name="update_status",
                allow_failure=True,
            )
        )

    after = observe()
    queue = build_self_improvement_queue(after, decision, teacher_result)
    write_json(QUEUE_PATH, queue)
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/sparkstream_history.py",
                "--append",
                "--out",
                "reports/sparkstream_history.json",
            ],
            timeout=120,
            execute=True,
            name="sparkstream_history_append",
        )
    )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/context_packet_ledger.py",
                "--ingest-reports",
                "--compact",
                "--max-active-packets",
                str(get_path(policy, ["context_packets", "max_active_packets"], 96)),
                "--max-active-chars",
                str(get_path(policy, ["context_packets", "max_active_chars"], 60000)),
                "--out",
                str(get_path(policy, ["context_packets", "report"], "reports/context_packet_ledger.json")),
            ],
            timeout=120,
            execute=True,
            name="context_packet_compaction",
            allow_failure=True,
        )
    )
    if get_path(policy, ["virtual_context_memory", "enabled"], False):
        commands.append(
            run_step(
                [
                    sys.executable,
                    str(get_path(policy, ["virtual_context_memory", "script"], "scripts/virtual_context_memory.py")),
                    "--context-ledger",
                    str(get_path(policy, ["context_packets", "report"], "reports/context_packet_ledger.json")),
                    "--packet-stream",
                    str(get_path(policy, ["context_packets", "ledger"], "reports/context_packets.jsonl")),
                    "--out-ledger",
                    str(get_path(policy, ["virtual_context_memory", "ledger"], "reports/virtual_context_memory_ledger.json")),
                    "--pages-out",
                    str(get_path(policy, ["virtual_context_memory", "pages"], "reports/virtual_context_memory_pages.jsonl")),
                    "--compiled-out",
                    str(get_path(policy, ["virtual_context_memory", "compiled_context"], "reports/virtual_context_compiled_context.json")),
                    "--probe-out",
                    str(get_path(policy, ["virtual_context_memory", "probe"], "reports/virtual_context_memory_probe.json")),
                    "--markdown-out",
                    str(get_path(policy, ["virtual_context_memory", "markdown_report"], "reports/virtual_context_memory_probe.md")),
                    "--event-log",
                    str(get_path(policy, ["virtual_context_memory", "event_log"], "reports/virtual_context_memory_events.jsonl")),
                    "--graph-out",
                    str(get_path(policy, ["virtual_context_memory", "graph"], "reports/virtual_context_memory_graph.json")),
                    "--transactions-out",
                    str(get_path(policy, ["virtual_context_memory", "transactions"], "reports/virtual_context_memory_transactions.jsonl")),
                    "--snapshots-out",
                    str(get_path(policy, ["virtual_context_memory", "snapshots"], "reports/virtual_context_memory_snapshots.json")),
                    "--bench-out",
                    str(get_path(policy, ["virtual_context_memory", "bench"], "reports/virtual_context_memory_bench.json")),
                    "--bench-markdown-out",
                    str(get_path(policy, ["virtual_context_memory", "bench_markdown"], "reports/virtual_context_memory_bench.md")),
                    "--index-out",
                    str(get_path(policy, ["virtual_context_memory", "index"], "reports/virtual_context_memory_index.json")),
                    "--usage-events",
                    str(get_path(policy, ["virtual_context_memory", "usage_events"], "reports/virtual_context_memory_usage_events.jsonl")),
                    "--training-admission-out",
                    str(get_path(policy, ["virtual_context_memory", "training_admission"], "reports/virtual_context_memory_training_admission.json")),
                    "--consumer-audit-out",
                    str(get_path(policy, ["virtual_context_memory", "consumer_audit"], "reports/virtual_context_memory_consumer_audit.json")),
                    "--token-budget",
                    str(get_path(policy, ["virtual_context_memory", "token_budget"], 6000)),
                    "--task",
                    str(get_path(policy, ["virtual_context_memory", "task"], "Project Theseus autonomy context and memory compilation")),
                ],
                timeout=180,
                execute=True,
                name="virtual_context_memory_refresh",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    str(get_path(policy, ["virtual_context_memory", "task_context_bridge_script"], "scripts/vcm_task_context_bridge.py")),
                    "--policy",
                    str(get_path(policy, ["virtual_context_memory", "task_context_policy"], "configs/vcm_task_context_policy.json")),
                    "--index",
                    str(get_path(policy, ["virtual_context_memory", "index"], "reports/virtual_context_memory_index.json")),
                    "--compiled",
                    str(get_path(policy, ["virtual_context_memory", "compiled_context"], "reports/virtual_context_compiled_context.json")),
                    "--probe",
                    str(get_path(policy, ["virtual_context_memory", "probe"], "reports/virtual_context_memory_probe.json")),
                    "--status",
                    str(get_path(policy, ["virtual_context_memory", "status"], "reports/virtual_context_memory_status.json")),
                    "--training-admission",
                    str(get_path(policy, ["virtual_context_memory", "training_admission"], "reports/virtual_context_memory_training_admission.json")),
                    "--consumer-audit",
                    str(get_path(policy, ["virtual_context_memory", "consumer_audit"], "reports/virtual_context_memory_consumer_audit.json")),
                    "--runtime-readiness",
                    str(get_path(policy, ["virtual_context_memory", "runtime_claim_readiness"], "reports/vcm_runtime_claim_readiness.json")),
                    "--release-conformance",
                    str(get_path(policy, ["virtual_context_memory", "release_conformance_audit"], "reports/vcm_release_conformance_audit.json")),
                    "--out",
                    str(get_path(policy, ["virtual_context_memory", "task_context_bridge"], "reports/vcm_task_context_bridge.json")),
                    "--markdown-out",
                    str(get_path(policy, ["virtual_context_memory", "task_context_bridge_markdown"], "reports/vcm_task_context_bridge.md")),
                    "--contexts-out",
                    str(get_path(policy, ["virtual_context_memory", "task_contexts"], "reports/vcm_task_contexts.json")),
                ],
                timeout=60,
                execute=True,
                name="vcm_task_context_bridge",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    str(get_path(policy, ["virtual_context_memory", "script"], "scripts/virtual_context_memory.py")),
                    "status",
                    "--out",
                    str(get_path(policy, ["virtual_context_memory", "status"], "reports/virtual_context_memory_status.json")),
                ],
                timeout=60,
                execute=True,
                name="virtual_context_memory_status",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    str(get_path(policy, ["virtual_context_memory", "context_recovery_script"], "scripts/vcm_context_recovery_benchmark.py")),
                    "--out",
                    str(get_path(policy, ["virtual_context_memory", "context_recovery_benchmark"], "reports/vcm_context_recovery_benchmark.json")),
                    "--markdown-out",
                    str(get_path(policy, ["virtual_context_memory", "context_recovery_markdown"], "reports/vcm_context_recovery_benchmark.md")),
                    "--residuals-out",
                    str(get_path(policy, ["virtual_context_memory", "context_recovery_residuals"], "reports/vcm_context_recovery_residuals.jsonl")),
                    "--vcm-probe",
                    str(get_path(policy, ["virtual_context_memory", "probe"], "reports/virtual_context_memory_probe.json")),
                    "--vcm-status",
                    str(get_path(policy, ["virtual_context_memory", "status"], "reports/virtual_context_memory_status.json")),
                    "--vcm-index",
                    str(get_path(policy, ["virtual_context_memory", "index"], "reports/virtual_context_memory_index.json")),
                    "--token-budget",
                    str(get_path(policy, ["virtual_context_memory", "token_budget"], 6000)),
                ],
                timeout=90,
                execute=True,
                name="vcm_context_recovery_benchmark",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    str(get_path(policy, ["virtual_context_memory", "prefetch_regret_script"], "scripts/vcm_prefetch_regret_audit.py")),
                    "--compiled",
                    str(get_path(policy, ["virtual_context_memory", "compiled_context"], "reports/virtual_context_compiled_context.json")),
                    "--usage-events",
                    str(get_path(policy, ["virtual_context_memory", "usage_events"], "reports/virtual_context_memory_usage_events.jsonl")),
                    "--out",
                    str(get_path(policy, ["virtual_context_memory", "prefetch_regret_audit"], "reports/vcm_prefetch_regret_audit.json")),
                    "--markdown-out",
                    str(get_path(policy, ["virtual_context_memory", "prefetch_regret_markdown"], "reports/vcm_prefetch_regret_audit.md")),
                ],
                timeout=60,
                execute=True,
                name="vcm_prefetch_regret_audit",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    str(get_path(policy, ["virtual_context_memory", "runtime_claim_readiness_script"], "scripts/vcm_runtime_claim_readiness.py")),
                    "--compiled",
                    str(get_path(policy, ["virtual_context_memory", "compiled_context"], "reports/virtual_context_compiled_context.json")),
                    "--pages",
                    str(get_path(policy, ["virtual_context_memory", "pages"], "reports/virtual_context_memory_pages.jsonl")),
                    "--out",
                    str(get_path(policy, ["virtual_context_memory", "runtime_claim_readiness"], "reports/vcm_runtime_claim_readiness.json")),
                    "--markdown-out",
                    str(get_path(policy, ["virtual_context_memory", "runtime_claim_readiness_markdown"], "reports/vcm_runtime_claim_readiness.md")),
                    "--claims-out",
                    str(get_path(policy, ["virtual_context_memory", "runtime_materialization_claims"], "reports/vcm_runtime_materialization_claims.jsonl")),
                ],
                timeout=60,
                execute=True,
                name="vcm_runtime_claim_readiness",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    str(get_path(policy, ["virtual_context_memory", "release_conformance_script"], "scripts/vcm_release_conformance_audit.py")),
                    "--out",
                    str(get_path(policy, ["virtual_context_memory", "release_conformance_audit"], "reports/vcm_release_conformance_audit.json")),
                    "--markdown-out",
                    str(get_path(policy, ["virtual_context_memory", "release_conformance_markdown"], "reports/vcm_release_conformance_audit.md")),
                ],
                timeout=60,
                execute=True,
                name="vcm_release_conformance_audit",
                allow_failure=True,
            )
        )
    for command in commands:
        if command.get("returncode", 0) != 0 and not command.get("allow_failure"):
            ok = False

    report = {
        "policy": "sparkstream_autonomy_cycle_v0",
        "cycle_id": cycle_id,
        "created_utc": now(),
        "requested_profile": requested_profile,
        "profile": profile,
        "execute": args.execute,
        "allow_teacher": teacher_allowed,
        "allow_network_fetch": network_allowed,
        "offline": bool(args.offline),
        "teacher_forbidden": teacher_forbidden,
        "network_fetch_forbidden": network_forbidden,
        "ok": ok,
        "decision": decision,
        "teacher_needed": teacher_needed,
        "teacher_used": bool(teacher_allowed and teacher_needed),
        "commands": commands,
        "before": compact_observation(before),
        "after": compact_observation(after),
        "self_improvement_queue": queue,
    }
    consume_watchdog_override(decision)
    consume_frontier_rotation_request(decision)
    write_json(ROOT / args.out, report)
    append_jsonl(LEDGER_PATH, compact_ledger_entry(report))
    update_status(cycle_id, "idle", profile, "Cycle complete.", ok=ok)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1




if __name__ == "__main__":
    raise SystemExit(main())
