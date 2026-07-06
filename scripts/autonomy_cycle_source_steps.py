"""Source, pantry, and training-inventory command assembly for autonomy cycles."""

from __future__ import annotations

import sys
from typing import Any

from autonomy_cycle_runtime import get_path, run_step, update_status


def append_source_and_training_inventory_steps(
    *,
    commands: list[dict[str, Any]],
    policy: dict[str, Any],
    decision: dict[str, Any],
    network_allowed: bool,
    policy_path: str,
    cycle_id: str,
    profile: str,
) -> None:
    update_status(cycle_id, "benchmark_seeker", profile, "Refreshing benchmark and data inventory.")
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/benchmark_seeker.py",
                "--refresh-local",
                "--out",
                "reports/benchmark_seeker_registry.json",
            ],
            timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
            execute=True,
            name="benchmark_seeker_refresh",
        )
    )
    if network_allowed and decision.get("benchmark_discovery_queries"):
        for index, query in enumerate(decision.get("benchmark_discovery_queries") or []):
            commands.append(
                run_step(
                    [
                        sys.executable,
                        "scripts/benchmark_seeker.py",
                        "--allow-network-discovery",
                        "--discover-query",
                        str(query),
                        "--discover-limit",
                        str(get_path(policy, ["benchmark_discovery", "discover_limit"], 10)),
                        "--out",
                        "reports/benchmark_seeker_registry.json",
                    ],
                    timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
                    execute=True,
                    name=f"benchmark_seeker_discovery_{index + 1}",
                    allow_failure=True,
                )
            )
    commands.append(
        run_step(
            [
                sys.executable,
                "scripts/knowledge_source_lookup.py",
                "--list",
                "--out",
                "reports/knowledge_source_registry.json",
            ],
            timeout=120,
            execute=True,
            name="knowledge_source_registry",
            allow_failure=True,
        )
    )
    online_source_command = [
        sys.executable,
        "scripts/online_source_catalog.py",
        "--catalog",
        str(get_path(policy, ["online_source_catalog", "catalog"], "configs/online_source_catalog.json")),
        "--out",
        str(get_path(policy, ["online_source_catalog", "report"], "reports/online_source_catalog_report.json")),
    ]
    if network_allowed and get_path(policy, ["online_source_catalog", "autonomous_import_allowed"], False):
        online_source_command.extend(
            [
                "--allow-network-fetch",
                "--import-sources",
                "--max-imports",
                str(get_path(policy, ["online_source_catalog", "max_imports_per_cycle"], 4)),
            ]
        )
    commands.append(
        run_step(
            online_source_command,
            timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
            execute=True,
            name="online_source_catalog",
            allow_failure=True,
        )
    )
    resource_pantry = policy.get("resource_pantry") or {}
    if resource_pantry.get("enabled", True):
        resource_pantry_command = [
            sys.executable,
            "scripts/resource_pantry.py",
            "--policy",
            str(resource_pantry.get("policy", "configs/resource_pantry.json")),
            "--catalog",
            str(resource_pantry.get("catalog", get_path(policy, ["online_source_catalog", "catalog"], "configs/online_source_catalog.json"))),
            "--out",
            str(resource_pantry.get("report", "reports/resource_pantry.json")),
            "--markdown-out",
            str(resource_pantry.get("markdown_report", "reports/resource_pantry.md")),
        ]
        if network_allowed and resource_pantry.get("execute_clone_when_allowed", False):
            resource_pantry_command.extend(
                [
                    "--execute",
                    "--max-clones",
                    str(resource_pantry.get("max_clones_per_cycle", 4)),
                ]
            )
        commands.append(
            run_step(
                resource_pantry_command,
                timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
                execute=True,
                name="resource_pantry",
                allow_failure=True,
            )
        )
    legacy_concepts = policy.get("legacy_concepts") or {}
    if legacy_concepts.get("enabled", True) and legacy_concepts.get("refresh_each_cycle", True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/legacy_project_concept_audit.py",
                    "--map",
                    str(legacy_concepts.get("map", "configs/legacy_concept_port_map.json")),
                    "--out",
                    str(legacy_concepts.get("report", "reports/legacy_project_concept_audit.json")),
                    "--markdown-out",
                    str(legacy_concepts.get("markdown_report", "reports/legacy_project_concept_audit.md")),
                ],
                timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
                execute=True,
                name="legacy_project_concept_audit",
                allow_failure=True,
            )
        )
    legacy_ports = policy.get("legacy_ports") or {}
    if legacy_ports.get("enabled", True) and legacy_ports.get("refresh_each_cycle", True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/legacy_port_mechanisms.py",
                    "--policy",
                    str(legacy_ports.get("policy", "configs/legacy_port_policy.json")),
                    "--out",
                    str(legacy_ports.get("report", "reports/legacy_port_mechanisms.json")),
                    "--markdown-out",
                    str(legacy_ports.get("markdown_report", "reports/legacy_port_mechanisms.md")),
                ],
                timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
                execute=True,
                name="legacy_port_mechanisms",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/high_transfer_curriculum_scheduler.py",
                    "--out",
                    "reports/high_transfer_curriculum_scheduler.json",
                    "--markdown-out",
                    "reports/high_transfer_curriculum_scheduler.md",
                    "--tasks-out",
                    "reports/high_transfer_curriculum_tasks.jsonl",
                ],
                timeout=120,
                execute=True,
                name="high_transfer_curriculum_scheduler",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/legacy_runtime_governance_gate.py",
                    "--out",
                    "reports/legacy_runtime_governance_gate.json",
                ],
                timeout=120,
                execute=True,
                name="legacy_runtime_governance_gate",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/coherence_delirium_gate.py",
                    "--out",
                    "reports/coherence_delirium_gate.json",
                ],
                timeout=120,
                execute=True,
                name="coherence_delirium_gate",
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
                name="legacy_port_runtime_enforcement",
                allow_failure=True,
            )
        )
    old_project_registry = policy.get("old_project_registry_port") or {}
    if old_project_registry.get("enabled", True) and old_project_registry.get("refresh_each_cycle", True):
        old_registry_command = [
            sys.executable,
            "scripts/old_project_registry_port.py",
            "--policy",
            str(old_project_registry.get("policy", "configs/old_project_registry_port_policy.json")),
            "--out",
            str(old_project_registry.get("report", "reports/old_project_registry_port.json")),
            "--markdown-out",
            str(old_project_registry.get("markdown_report", "reports/old_project_registry_port.md")),
        ]
        if old_project_registry.get("write_cards", True):
            old_registry_command.append("--write-cards")
        commands.append(
            run_step(
                old_registry_command,
                timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
                execute=True,
                name="old_project_registry_port",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/legacy_training_source_audit.py",
                    "--sources",
                    "data/training_sources/old_project_registry_training_sources.json",
                    "--registry-report",
                    str(old_project_registry.get("report", "reports/old_project_registry_port.json")),
                    "--out",
                    "reports/legacy_training_source_audit.json",
                    "--markdown-out",
                    "reports/legacy_training_source_audit.md",
                    "--admissions-out",
                    "data/training_sources/legacy_training_admissions.json",
                ],
                timeout=180,
                execute=True,
                name="legacy_training_source_audit",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/legacy_training_source_sampler.py",
                    "--admissions",
                    "data/training_sources/legacy_training_admissions.json",
                    "--out",
                    "reports/legacy_training_source_sample.json",
                    "--sample-out",
                    "data/training_sources/legacy_tiny_dry_run_sample.jsonl",
                    "--max-rows",
                    "128",
                ],
                timeout=240,
                execute=True,
                name="legacy_training_source_sampler",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/trace_fabric_capsule_admission.py",
                    "--trace-exchange",
                    "reports/trace_fabric_training_exchange.json",
                    "--self-mod-proof",
                    "reports/self_mod_proof_bundle.json",
                    "--out",
                    "reports/trace_fabric_capsule_admission.json",
                    "--candidates-out",
                    "data/training_sources/trace_fabric_capsule_candidates.jsonl",
                ],
                timeout=180,
                execute=True,
                name="trace_fabric_capsule_admission",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/trace_fabric_capsule_materializer.py",
                    "--admission",
                    "reports/trace_fabric_capsule_admission.json",
                    "--candidates",
                    "data/training_sources/trace_fabric_capsule_candidates.jsonl",
                    "--out",
                    "reports/trace_fabric_capsule_materialization.json",
                    "--rows-out",
                    "data/training_sources/trace_fabric_materialized_training_rows.jsonl",
                    "--max-rows",
                    "64",
                ],
                timeout=180,
                execute=True,
                name="trace_fabric_capsule_materializer",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/legacy_adapter_bank_training_plan.py",
                    "--bank",
                    "reports/low_rank_adapter_bank.json",
                    "--legacy-sample",
                    "reports/legacy_training_source_sample.json",
                    "--trace-materialization",
                    "reports/trace_fabric_capsule_materialization.json",
                    "--taskspell",
                    "reports/taskspell_contracts.json",
                    "--runtime-governance",
                    "reports/legacy_runtime_governance_gate.json",
                    "--out",
                    "reports/legacy_adapter_bank_training_plan.json",
                    "--plan-out",
                    "data/training_sources/legacy_adapter_bank_dry_run_plan.jsonl",
                ],
                timeout=180,
                execute=True,
                name="legacy_adapter_bank_training_plan",
                allow_failure=True,
            )
        )
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/legacy_active_inference_pilot.py",
                    "--active-inference",
                    "reports/active_inference_world_model.json",
                    "--world-runtime",
                    "reports/world_adapter_job_runtime.json",
                    "--out",
                    "reports/legacy_active_inference_pilot.json",
                    "--belief-out",
                    "data/world_model/active_inference_belief_updates.jsonl",
                ],
                timeout=120,
                execute=True,
                name="legacy_active_inference_pilot",
                allow_failure=True,
            )
        )
    training_sampler_command = [
        sys.executable,
        "scripts/training_data_sampler.py",
        "--policy",
        str(policy_path),
        "--catalog",
        str(get_path(policy, ["online_source_catalog", "catalog"], "configs/online_source_catalog.json")),
        "--catalog-report",
        str(get_path(policy, ["online_source_catalog", "report"], "reports/online_source_catalog_report.json")),
        "--out",
        "reports/training_data_sampler.json",
    ]
    if network_allowed and get_path(policy, ["training_data", "autonomous_small_samples"], False):
        training_sampler_command.append("--allow-network-fetch")
    commands.append(
        run_step(
            training_sampler_command,
            timeout=int(get_path(policy, ["command_timeouts_seconds", "maintenance"], 1800)),
            execute=True,
            name="training_data_sampler",
            allow_failure=True,
        )
    )
    conversation_pantry = policy.get("open_conversation_training_pantry") or {}
    if conversation_pantry.get("enabled", True):
        conversation_command = [
            sys.executable,
            "scripts/open_conversation_training_pantry.py",
            "--config",
            str(conversation_pantry.get("config", "configs/open_conversation_training_pantry.json")),
            "--root",
            str(conversation_pantry.get("root", "D:/ProjectTheseus/training_data/open_conversation_pantry")),
            "--max-rows-per-source",
            str(conversation_pantry.get("max_rows_per_source", 24)),
            "--out",
            str(conversation_pantry.get("report", "reports/open_conversation_training_pantry.json")),
            "--markdown-out",
            str(conversation_pantry.get("markdown_report", "reports/open_conversation_training_pantry.md")),
        ]
        if network_allowed:
            conversation_command.append("--allow-network-fetch")
        commands.append(
            run_step(
                conversation_command,
                timeout=240,
                execute=True,
                name="open_conversation_training_pantry",
                allow_failure=True,
            )
        )
    long_horizon_programming = policy.get("long_horizon_programming") or {}
    if long_horizon_programming.get("enabled", True):
        commands.append(
            run_step(
                [
                    sys.executable,
                    "scripts/long_horizon_programming_curriculum.py",
                    "--task-out",
                    str(
                        long_horizon_programming.get(
                            "task_out",
                            "D:/ProjectTheseus/training_data/long_horizon_programming/private_train/repo_repair_tasks.jsonl",
                        )
                    ),
                    "--sts-out",
                    str(
                        long_horizon_programming.get(
                            "sts_out",
                            "D:/ProjectTheseus/training_data/long_horizon_programming/sts/repo_repair_sts_rows.jsonl",
                        )
                    ),
                    "--out",
                    str(long_horizon_programming.get("report", "reports/long_horizon_programming_curriculum.json")),
                    "--markdown-out",
                    str(
                        long_horizon_programming.get(
                            "markdown_report",
                            "reports/long_horizon_programming_curriculum.md",
                        )
                    ),
                ],
                timeout=120,
                execute=True,
                name="long_horizon_programming_curriculum",
                allow_failure=True,
            )
        )
