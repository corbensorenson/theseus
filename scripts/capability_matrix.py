"""Build the live SparkStream capability and market comparison matrix.

The matrix is intentionally evidence driven. Local status comes from generated
reports under reports/, while market baselines come from a small source-backed
config that should be refreshed before public claims.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "capability_market_baselines.json"
DEFAULT_OUT = ROOT / "reports" / "capability_matrix.json"
DEFAULT_MARKDOWN_OUT = ROOT / "reports" / "capability_matrix.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    config = read_json(ROOT / args.config, {})
    reports = load_reports()
    rows = build_rows(config, reports)
    summary = build_summary(config, reports, rows)
    payload = {
        "policy": "sparkstream_capability_matrix_v0",
        "created_utc": now(),
        "config": str(Path(args.config)).replace("\\", "/"),
        "summary": summary,
        "market_systems": config.get("market_systems", []),
        "matrix": rows,
        "source_reports": source_reports(),
        "refresh_guidance": {
            "local": "Regenerate each autonomy cycle and whenever benchmark, arm, data, or checkpoint reports change.",
            "market": config.get("refresh_policy", {}),
            "claim_safety": "Treat market comparisons as directional unless the source URLs were checked for the current public claim.",
        },
    }
    write_json(ROOT / args.out, payload)
    write_text(ROOT / args.markdown_out, render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0


def load_reports() -> dict[str, Any]:
    reports = ROOT / "reports"
    return {
        "architecture_gate": read_json(reports / "architecture_gate_report.json", {}),
        "arm_lifecycle": read_json(reports / "arm_lifecycle_governance.json", {}),
        "arm_registry": read_json(reports / "arm_registry.json", {}),
        "arm_sucker": read_json(reports / "arm_sucker_registry.json", {}),
        "autonomy_launch_readiness": read_json(reports / "autonomy_launch_readiness.json", {}),
        "attd": read_json(reports / "attd_report.json", {}),
        "attd_packets": read_json(reports / "attd_maintenance_packets.json", {}),
        "benchmark_ledger": read_json(reports / "benchmark_ledger.json", []),
        "benchmark_adapter_factory": read_json(reports / "benchmark_adapter_factory.json", {}),
        "candidate_bottleneck_reducer": read_json(reports / "candidate_bottleneck_reducer.json", {}),
        "benchmark_seeker": read_json(reports / "benchmark_seeker_registry.json", {}),
        "ai_grand_prix_spec": read_json(reports / "ai_grand_prix_spec_digest.json", {}),
        "minecraft_runtime_probe": read_json(reports / "minecraft_runtime_probe.json", {}),
        "candidate_gate": read_json(reports / "candidate_promotion_gate.json", {}),
        "promotion_closure": read_json(reports / "promotion_closure.json", {}),
        "accepted_candidate_registry": read_json(reports / "accepted_candidate_registry.json", {}),
        "checkpoint_registry": read_json(reports / "checkpoint_registry.json", {}),
        "context_packets": read_json(reports / "context_packet_ledger.json", {}),
        "virtual_context_memory": read_json(reports / "virtual_context_memory_probe.json", {}),
        "virtual_context_memory_bench": read_json(reports / "virtual_context_memory_bench.json", {}),
        "virtual_context_memory_index": read_json(reports / "virtual_context_memory_index.json", {}),
        "virtual_context_memory_status": read_json(reports / "virtual_context_memory_status.json", {}),
        "virtual_context_memory_training_admission": read_json(reports / "virtual_context_memory_training_admission.json", {}),
        "virtual_context_memory_consumer_audit": read_json(reports / "virtual_context_memory_consumer_audit.json", {}),
        "vcm_context_recovery_benchmark": read_json(reports / "vcm_context_recovery_benchmark.json", {}),
        "knowledge_sources": read_json(reports / "knowledge_source_registry.json", {}),
        "hive_status": read_json(reports / "hive_status.json", {}),
        "hive_peers": read_json(reports / "hive_peers.json", {}),
        "hive_scheduler": read_json(reports / "hive_scheduler.json", {}),
        "hive_relay": read_json(reports / "hive_relay_status.json", {}),
        "license_status": read_json(reports / "license_status.json", {}),
        "compute_market": read_json(reports / "compute_market_status.json", {}),
        "public_hive_contribution": read_json(reports / "public_hive_contribution_status.json", {}),
        "model_ledger": read_json(reports / "model_ledger.json", {}),
        "native_voice_io": read_json(reports / "native_voice_io.json", {}),
        "native_voice_training_manifest": read_json(reports / "native_voice_training_manifest.json", {}),
        "online_source_catalog": read_json(reports / "online_source_catalog_report.json", {}),
        "openai_compat": read_json(reports / "openai_compat_status.json", {}),
        "public_comparator": read_json(reports / "public_comparator_ledger.json", {}),
        "python_runtime": read_json(reports / "python_runtime_compatibility.json", {}),
        "architecture_experiment_runner": read_json(reports / "architecture_experiment_runner.json", {}),
        "arm_transfer_artifacts": read_json(reports / "arm_transfer_artifacts.json", {}),
        "teacher_self_edit_proof": read_json(reports / "teacher_self_edit_proof.json", {}),
        "resource_governor": read_json(reports / "resource_governor.json", {}),
        "performance_optimizer": read_json(reports / "performance_optimizer.json", {}),
        "legacy_port_mechanisms": read_json(reports / "legacy_port_mechanisms.json", {}),
        "planforge_schedule": read_json(reports / "planforge_schedule.json", {}),
        "coherence_delirium": read_json(reports / "coherence_delirium_report.json", {}),
        "proxy_truth_audit": read_json(reports / "proxy_truth_audit.json", {}),
        "world_adapter_jobs": read_json(reports / "world_adapter_job_runtime.json", {}),
        "emulator_game_trace_gateway": read_json(reports / "emulator_game_trace_gateway.json", {}),
        "active_inference_world_model": read_json(reports / "active_inference_world_model.json", {}),
        "macro_counterexample_gate": read_json(reports / "macro_counterexample_gate.json", {}),
        "bridge_adapter_native_promotion": read_json(reports / "bridge_adapter_native_promotion.json", {}),
        "pretraining_readiness_integrity": read_json(reports / "pretraining_readiness_integrity.json", {}),
        "salience_scheduler": read_json(reports / "salience_scheduler.json", {}),
        "campaign_dag": read_json(reports / "campaign_dag.json", {}),
        "dataset_recipe_scaffolder": read_json(reports / "dataset_recipe_scaffolder.json", {}),
        "evidence_graph_ledger": read_json(reports / "evidence_graph_ledger.json", {}),
        "runtime_resolution_boundary": read_json(reports / "runtime_resolution_boundary.json", {}),
        "tiered_memory_consolidation": read_json(reports / "tiered_memory_consolidation.json", {}),
        "aletheia_advocate_gate": read_json(reports / "aletheia_advocate_gate.json", {}),
        "synaptic_work_stealing": read_json(reports / "synaptic_work_stealing.json", {}),
        "architecture_motif_library": read_json(reports / "architecture_motif_library.json", {}),
        "semantic_intent_repair": read_json(reports / "semantic_intent_repair.json", {}),
        "eval_track_contract_library": read_json(reports / "eval_track_contract_library.json", {}),
        "synaptic_permission_decay": read_json(reports / "synaptic_permission_decay.json", {}),
        "temporal_replay_assertions": read_json(reports / "temporal_replay_assertions.json", {}),
        "whitecell_threat_memory": read_json(reports / "whitecell_threat_memory.json", {}),
        "zero_copy_context_prefetch": read_json(reports / "zero_copy_context_prefetch.json", {}),
        "hil_emulator_gate": read_json(reports / "hil_emulator_gate.json", {}),
        "formal_runtime_coupling": read_json(reports / "formal_runtime_coupling.json", {}),
        "veritas_discovery_novelty": read_json(reports / "veritas_discovery_novelty.json", {}),
        "anti_expert_tribunal_router": read_json(reports / "anti_expert_tribunal_router.json", {}),
        "probe_router_burst_budget": read_json(reports / "probe_router_burst_budget.json", {}),
        "rlds_minari_trace_export": read_json(reports / "rlds_minari_trace_export.json", {}),
        "live_operator_advisors": read_json(reports / "live_operator_advisors.json", {}),
        "benchmark_bounty_registry": read_json(reports / "benchmark_bounty_registry.json", {}),
        "legacy_fine_tooth_comb": read_json(reports / "legacy_fine_tooth_comb.json", {}),
        "rl_registry": read_json(reports / "rl_benchmark_registry.json", {}),
        "rmi": read_json(reports / "ratcheting_modular_intelligence_report.json", {}),
        "router": read_json(reports / "octopus_router_report.json", {}),
        "router_head": read_json(reports / "octopus_router_head_report.json", {}),
        "safety_ledger": read_json(reports / "safety_benchmark_ledger.json", {}),
        "self_improvement_queue": read_json(reports / "self_improvement_queue.json", {}),
        "sparkstream_history": read_json(reports / "sparkstream_history.json", {}),
        "synthetic_data": read_json(reports / "synthetic_data_curator.json", {}),
        "tool_registry": read_json(reports / "tool_registry.json", {}),
        "training_data_inventory": read_json(reports / "training_data_inventory.json", {}),
        "update_status": read_json(reports / "update_status.json", {}),
        "update_offer": read_json(reports / "update_offer_current.json", {}),
    }


def build_rows(config: dict[str, Any], reports: dict[str, Any]) -> list[dict[str, Any]]:
    domains = {row.get("id"): row for row in config.get("capability_domains", []) if isinstance(row, dict)}
    market = {row.get("id"): row for row in config.get("market_systems", []) if isinstance(row, dict)}
    builders = {
        "rmi_architecture_governance": rmi_architecture_row,
        "octopus_router_arms": octopus_router_row,
        "arm_sucker_transfer_hierarchy": arm_sucker_transfer_row,
        "benchmark_ratchet": benchmark_ratchet_row,
        "public_calibration_market_comparison": public_calibration_row,
        "loop_closure_tool_registry": loop_closure_row,
        "autonomous_daemon_teacher": autonomous_daemon_row,
        "dashboard_observability": dashboard_row,
        "distributed_hive_app_runtime": hive_runtime_row,
        "checkpoint_versioning_chat": checkpoint_row,
        "resource_aware_rust_cuda_training": resource_row,
        "language_grammar_babylm": language_row,
        "rl_environment_learning": rl_row,
        "minecraft_open_world_rl": minecraft_open_world_row,
        "drone_racing_control": drone_racing_control_row,
        "native_voice_io": native_voice_io_row,
        "training_data_governance": training_data_row,
        "synthetic_data_curation": synthetic_data_row,
        "context_packet_memory": context_packet_row,
        "safety_permissions_reflex": safety_row,
        "attd_repo_health_governance": attd_row,
        "codebase_engineering_agent": codebase_engineering_row,
        "external_research_and_benchmark_discovery": external_research_row,
    }
    rows: list[dict[str, Any]] = []
    for domain in config.get("capability_domains", []):
        domain_id = domain.get("id")
        builder = builders.get(domain_id, generic_row)
        rows.append(builder(domain, reports, market))
    rows.append(capability_matrix_row(reports, market))
    return rows


def rmi_architecture_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    score = number(get_path(reports, ["rmi", "implementation_score", "score"], 0.0))
    implemented = int(number(get_path(reports, ["rmi", "implementation_score", "implemented"], 0)))
    possible = int(number(get_path(reports, ["rmi", "implementation_score", "possible"], 0)))
    gate_green = get_path(reports, ["architecture_gate", "green"], None)
    maturity = clamp(0.55 + score * 0.35 + (0.1 if gate_green is True else 0.0))
    status = "ready" if score >= 0.95 and gate_green is not False else "active"
    return make_row(
        domain,
        status=status,
        maturity=maturity,
        market_position="differentiated_research",
        evidence=[
            f"RMI implementation score {score:.3f}",
            f"{implemented}/{possible} RMI components implemented",
            evidence_bool("architecture gate green", gate_green),
        ],
        gaps=["Keep architecture gate current after every major change.", "Prove durability with multi-week autonomous runs."],
        next_actions=["Refresh RMI and architecture-gate reports during every autonomy cycle."],
        confidence=0.88 if score else 0.5,
        market=market,
    )


def octopus_router_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    arms = get_path(reports, ["arm_registry", "arms"], [])
    arm_count = len(arms) if isinstance(arms, list) else 0
    exact = number(
        get_path(
            reports,
            ["router_head", "metrics", "exact_set_accuracy"],
            get_path(
                reports,
                ["router_head", "evaluation", "exact_set_accuracy"],
                get_path(reports, ["router_head", "learned_router", "exact_set_accuracy"], 0.0),
            ),
        )
    )
    governance_ready = bool(get_path(reports, ["arm_lifecycle", "ready_for_long_autonomy"], False))
    proposals = int(number(get_path(reports, ["arm_lifecycle", "summary", "proposal_count"], 0)))
    maturity = clamp(0.2 + min(arm_count / 12.0, 1.0) * 0.3 + exact * 0.25 + (0.2 if governance_ready else 0.0))
    return make_row(
        domain,
        status="ready" if arm_count >= 8 and governance_ready else "active",
        maturity=maturity,
        market_position="differentiated_research",
        evidence=[
            f"{arm_count} registered arms",
            f"learned router exact-set accuracy {exact:.3f}",
            evidence_bool("arm lifecycle ready for long autonomy", governance_ready),
            f"{proposals} lifecycle proposals/watch items",
        ],
        gaps=["Lifecycle manager is report-only; automated split/merge still requires reviewed implementation.", "Router trace set is still small compared with commercial-scale agent telemetry."],
        next_actions=["Continue appending real routing traces.", "Promote mature lifecycle proposals into reviewed migrations."],
        confidence=0.84,
        market=market,
    )


def arm_sucker_transfer_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    registry = reports.get("arm_sucker") if isinstance(reports.get("arm_sucker"), dict) else {}
    summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
    cores = int(number(summary.get("core_count", 0)))
    suckers = int(number(summary.get("sucker_count", 0)))
    ready = int(number(summary.get("ready_suckers", 0)))
    blocked = int(number(summary.get("blocked_suckers", 0)))
    maturity_avg = number(summary.get("average_sucker_maturity", 0.0))
    ready_for_routing = bool(summary.get("ready_for_transfer_routing"))
    top_ready = summary.get("top_ready_suckers") if isinstance(summary.get("top_ready_suckers"), list) else []
    maturity = clamp(0.15 + min(cores / 3.0, 1.0) * 0.2 + min(ready / 6.0, 1.0) * 0.25 + maturity_avg * 0.25 + (0.1 if ready_for_routing else 0.0))
    status = "ready" if ready_for_routing and blocked == 0 and ready else ("active" if ready else "partial")
    return make_row(
        domain,
        status=status,
        maturity=maturity,
        market_position="differentiated_research",
        evidence=[
            f"{cores} high-transfer cores",
            f"{ready}/{suckers} suckers ready",
            f"average sucker maturity {maturity_avg:.3f}",
            f"top ready suckers: {', '.join(str(item) for item in top_ready[:4]) or 'none'}",
        ],
        gaps=[
            "Blocked suckers still need required reports or artifacts." if blocked else "Sibling-transfer quality still needs repeated proof across games/environments.",
            "Sucker promotion into a new arm should require repeated transfer beyond the parent family.",
        ],
        next_actions=[
            "Route game-specific pressure through video_game_play_arm suckers before spawning new game arms.",
            "Require graduated suckers to export reusable priors, traces, and residual curricula.",
        ],
        confidence=0.78 if registry else 0.45,
        market=market,
    )


def benchmark_ratchet_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    ledger = reports.get("benchmark_ledger") if isinstance(reports.get("benchmark_ledger"), list) else []
    frontiers = [row for row in ledger if row.get("lifecycle") == "frontier"]
    regressions = [row for row in ledger if row.get("lifecycle") == "regression"]
    residual_summary = get_path(reports, ["tool_registry", "residual_escrow_summary"], {})
    residual_clusters = int(number(get_path(reports, ["tool_registry", "residual_escrow_summary", "clusters"], get_path(reports, ["rmi", "five_pillars", "active_compression", "residual_clusters"], 0))))
    candidate = reports.get("candidate_gate") if isinstance(reports.get("candidate_gate"), dict) else {}
    promotion_closure = reports.get("promotion_closure") if isinstance(reports.get("promotion_closure"), dict) else {}
    accepted_count = len(get_path(reports, ["accepted_candidate_registry", "accepted_candidates"], []))
    passed = int(number(candidate.get("passed", 0)))
    total = int(number(candidate.get("total", 0)))
    gate_fraction = passed / total if total else 0.0
    frontier_floor_ok = all(number(row.get("score")) >= number(get_path(row, ["graduation_policy", "floor_threshold"], 0.0)) for row in frontiers) if frontiers else True
    maturity = clamp(0.25 + min(len(ledger) / 12.0, 1.0) * 0.2 + min(len(regressions) / 10.0, 1.0) * 0.2 + gate_fraction * 0.2 + (0.1 if residual_clusters else 0.0))
    status = "active"
    if not ledger:
        status = "blocked"
    elif frontiers and not frontier_floor_ok:
        status = "partial"
    elif len(regressions) >= 8 and residual_clusters:
        status = "ready"
    return make_row(
        domain,
        status=status,
        maturity=maturity,
        market_position="differentiated_research",
        evidence=[
            f"{len(ledger)} benchmark ledger entries",
            f"{len(frontiers)} frontier / {len(regressions)} regression",
            f"candidate gate {passed}/{total}",
            f"promotion closure {promotion_closure.get('status', 'missing')} / accepted candidates {accepted_count}",
            f"residual clusters {residual_clusters}",
        ],
        gaps=["The active frontier must stay non-contaminated and genuinely diagnostic.", "Keep proving promotion -> checkpoint -> backup -> rotate over repeated cycles."],
        next_actions=["Verify every promotion writes a lifecycle override and rotation request.", "Keep residual deltas in the candidate gate."],
        confidence=0.86,
        market=market,
    )


def public_calibration_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    public = reports.get("public_comparator") or {}
    market_config_present = (ROOT / "configs" / "capability_market_baselines.json").exists()
    comparators = get_path(public, ["comparators"], [])
    count = len(comparators) if isinstance(comparators, list) else int(bool(public))
    maturity = clamp(0.35 + (0.25 if market_config_present else 0.0) + min(count / 3.0, 1.0) * 0.25)
    return make_row(
        domain,
        status="active" if market_config_present else "partial",
        maturity=maturity,
        market_position="catching_up",
        evidence=[
            evidence_bool("market baseline config present", market_config_present),
            f"{count} public comparator records",
            "Capability matrix generated from local reports and source-backed market notes.",
        ],
        gaps=["Add periodic web/source refresh automation before public-facing comparisons.", "Add external benchmark score comparators beyond current BabyLM/public probes."],
        next_actions=["Refresh market baselines weekly or before public release.", "Add measured market-facing evals when local task domains expand."],
        confidence=0.72,
        market=market,
    )


def loop_closure_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    tools = get_path(reports, ["tool_registry", "tools"], [])
    tool_count = len(tools) if isinstance(tools, list) else 0
    health = get_path(reports, ["tool_registry", "registry_health"], {})
    active = int(number(health.get("active", 0)))
    proposed = int(number(health.get("proposed", 0)))
    maturity = clamp(0.25 + min(active / 20.0, 1.0) * 0.45 + (0.15 if tool_count else 0.0))
    return make_row(
        domain,
        status="active" if active else "partial",
        maturity=maturity,
        market_position="differentiated_research",
        evidence=[f"{tool_count} tool cards", f"{active} active / {proposed} proposed tools", "Tool cards include preconditions, postconditions, risk, verification, and retirement fields."],
        gaps=["Need more runtime monitoring from real repeated workflows.", "Automated tool synthesis remains governed and mostly report-driven."],
        next_actions=["Use real autonomy-cycle traces to detect recurring workflows.", "Retire or consolidate stale tools as usage data accumulates."],
        confidence=0.78,
        market=market,
    )


def autonomous_daemon_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    launch = reports.get("autonomy_launch_readiness") if isinstance(reports.get("autonomy_launch_readiness"), dict) else {}
    ready = bool(launch.get("ready_for_autonomous_training"))
    teacher_ready = bool(launch.get("ready_for_teacher_enabled_run"))
    queue_items = get_path(reports, ["self_improvement_queue", "items"], [])
    queue_count = len(queue_items) if isinstance(queue_items, list) else 0
    teacher_proof = reports.get("teacher_self_edit_proof") if isinstance(reports.get("teacher_self_edit_proof"), dict) else {}
    teacher_success_rate = number(get_path(teacher_proof, ["summary", "success_rate"], 0.0))
    architecture_runner = reports.get("architecture_experiment_runner") if isinstance(reports.get("architecture_experiment_runner"), dict) else {}
    runner_ready = architecture_runner.get("status") in {"planned", "completed", "no_safe_experiment_selected"}
    maturity = clamp(
        0.25
        + (0.22 if ready else 0.0)
        + (0.17 if teacher_ready else 0.0)
        + min(queue_count / 25.0, 1.0) * 0.12
        + min(teacher_success_rate, 1.0) * 0.12
        + (0.07 if runner_ready else 0.0)
    )
    return make_row(
        domain,
        status="active" if ready else "partial",
        maturity=maturity,
        market_position="behind_products_but_differentiated",
        evidence=[
            evidence_bool("autonomous training ready", ready),
            evidence_bool("teacher-enabled run ready", teacher_ready),
            f"{queue_count} self-improvement queue items",
            f"teacher self-edit proof {teacher_proof.get('status', 'missing')} success_rate={teacher_success_rate:.2f}",
            f"architecture experiment runner {architecture_runner.get('status', 'missing')}",
        ],
        gaps=["Long-running reliability is not yet proven over many unattended days.", "Guarded teacher edits still need repeated boring successes before trusting larger changes."],
        next_actions=["Run controlled overnight and multi-day daemon tests.", "Keep architecture experiments matched and automatically ledgered."],
        confidence=0.76,
        market=market,
    )


def dashboard_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    files = ["scripts/sparkstream_dashboard.py", "dashboard/index.html", "dashboard/app.js"]
    present = [path for path in files if (ROOT / path).exists()]
    history_points = int(number(get_path(reports, ["sparkstream_history", "summary", "points"], 0)))
    openai_compat = reports.get("openai_compat") if isinstance(reports.get("openai_compat"), dict) else {}
    update_status = reports.get("update_status") if isinstance(reports.get("update_status"), dict) else {}
    compat_configured = bool(openai_compat.get("base_url"))
    update_ready = bool(update_status.get("policy") == "project_theseus_update_status_v0")
    maturity = clamp(0.38 + len(present) / len(files) * 0.32 + min(history_points / 25.0, 1.0) * 0.14 + (0.08 if compat_configured else 0.0) + (0.08 if update_ready else 0.0))
    return make_row(
        domain,
        status="ready" if len(present) == len(files) else "partial",
        maturity=maturity,
        market_position="parity_for_local_research",
        evidence=[
            f"{len(present)}/{len(files)} dashboard files present",
            f"{history_points} historical metric points",
            "Status API streams reports and active jobs.",
            evidence_bool("OpenAI-compatible local endpoint configured", compat_configured),
            f"OpenAI-compatible endpoint live: {'yes' if openai_compat.get('live') else 'no'}",
            evidence_bool("accepted-candidate update channel ready", update_ready),
            f"Update available: {'yes' if update_status.get('update_available') else 'no'}",
        ],
        gaps=["Dashboard is local research UI, not a hardened multi-user product.", "Add more charts as new domains mature."],
        next_actions=["Keep capability matrix visible in dashboard.", "Add per-capability trend history after multiple cycles.", "Use the local OpenAI-compatible endpoint for OpenCode/Hermes-style harness smoke tests."],
        confidence=0.83,
        market=market,
    )


def hive_runtime_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    status = reports.get("hive_status") if isinstance(reports.get("hive_status"), dict) else {}
    peers = reports.get("hive_peers") if isinstance(reports.get("hive_peers"), dict) else {}
    scheduler = reports.get("hive_scheduler") if isinstance(reports.get("hive_scheduler"), dict) else {}
    relay = reports.get("hive_relay") if isinstance(reports.get("hive_relay"), dict) else {}
    license_status = reports.get("license_status") if isinstance(reports.get("license_status"), dict) else {}
    compute_market = reports.get("compute_market") if isinstance(reports.get("compute_market"), dict) else {}
    public_contribution = reports.get("public_hive_contribution") if isinstance(reports.get("public_hive_contribution"), dict) else {}
    policy_exists = (ROOT / "configs" / "hive_policy.json").exists()
    node_ok = status.get("policy") == "project_theseus_hive_node_status_v0"
    scheduler_ok = scheduler.get("policy") == "project_theseus_hive_scheduler_v0"
    relay_ok = relay.get("policy") == "project_theseus_hive_relay_status_v0"
    setup_wizard_ok = (ROOT / "scripts" / "theseus_setup_wizard.py").exists()
    profiles_ok = (ROOT / "scripts" / "hive_profiles.py").exists()
    worker_chunk_ok = (ROOT / "scripts" / "hive_worker_chunk.py").exists()
    public_contribution_ok = public_contribution.get("policy") == "theseus_hive_public_contribution_status_v0"
    peer_count = int(number(peers.get("peer_count", 0)))
    real_worker_chunks = int(number(get_path(scheduler, ["summary", "real_worker_chunks"], 0)))
    license_complete = bool(license_status.get("registration_complete"))
    worker_chunks_licensed = bool(get_path(license_status, ["feature_summary", "can_run_worker_chunks"], False))
    company_hive_licensed = bool(get_path(license_status, ["feature_summary", "can_create_company_hive"], False))
    public_gateway_licensed = bool(get_path(license_status, ["feature_summary", "can_operate_public_gateway"], False))
    market_ok = compute_market.get("policy") == "project_theseus_compute_market_status_v0"
    market_balance = int(number(get_path(compute_market, ["balances", "available_micro_twc"], 0)))
    capabilities = status.get("capabilities") if isinstance(status.get("capabilities"), list) else []
    cap_ids = {cap.get("id") for cap in capabilities if isinstance(cap, dict)}
    has_accelerator = bool({"nvidia_cuda", "apple_mlx", "mlx_apple", "mlx_cuda"} & cap_ids)
    maturity = clamp(
        0.2
        + (0.18 if policy_exists else 0.0)
        + (0.18 if node_ok else 0.0)
        + (0.18 if scheduler_ok else 0.0)
        + (0.08 if relay_ok else 0.0)
        + (0.07 if setup_wizard_ok else 0.0)
        + (0.05 if profiles_ok else 0.0)
        + (0.08 if worker_chunk_ok and real_worker_chunks else 0.0)
        + (0.05 if license_complete and worker_chunks_licensed else 0.0)
        + (0.05 if public_contribution_ok else 0.0)
        + (0.06 if market_ok else 0.0)
        + min(peer_count / 3.0, 1.0) * 0.14
        + (0.12 if has_accelerator else 0.0)
    )
    status_label = "ready" if node_ok and scheduler_ok and peer_count else ("active" if node_ok and scheduler_ok else "partial")
    return make_row(
        domain,
        status=status_label,
        maturity=maturity,
        market_position="differentiated_local_runtime",
        evidence=[
            evidence_bool("Hive policy present", policy_exists),
            evidence_bool("local node probe available", node_ok),
            evidence_bool("scheduler report available", scheduler_ok),
            evidence_bool("relay report available", relay_ok),
            evidence_bool("no-terminal setup wizard available", setup_wizard_ok),
            evidence_bool("Hive profile switching available", profiles_ok),
            evidence_bool("license registration complete", license_complete),
            evidence_bool("worker chunks licensed", worker_chunks_licensed),
            evidence_bool("company Hive licensed", company_hive_licensed),
            evidence_bool("public gateway licensed", public_gateway_licensed),
            evidence_bool("CUDA/MLX worker chunk runner available", worker_chunk_ok),
            f"{real_worker_chunks} real worker chunk placements",
            evidence_bool("public contribution bridge available", public_contribution_ok),
            evidence_bool("compute market accounting available", market_ok),
            f"work-credit balance {market_balance} micro TWC",
            f"public contribution mode {public_contribution.get('mode', 'unknown')}",
            f"{peer_count} discovered peers",
            f"capabilities {', '.join(sorted(str(cap) for cap in cap_ids if cap)) or 'none'}",
        ],
        gaps=[
            "Native signed installers are not packaged yet; current click flow is a local wizard plus launcher scripts.",
            "Remote task execution is intentionally limited to registered task kinds and requires a shared secret off loopback.",
            "Public contribution is worker-only and still blocked from real public training shards until signed manifests, sandboxing, reputation, and quotas are live.",
            "Compute market is internal accounting only; public exchange/token operation is disabled until legal, custody, anti-fraud, and signed-worker controls exist.",
            "Distributed gradient/model-state synchronization is intentionally deferred; current hive routing dispatches arm-aware CUDA/MLX jobs and merges artifacts/reports rather than sharing optimizer state.",
        ],
        next_actions=[
            "Build signed Windows/macOS/Linux installer artifacts from the supervisor packaging scaffold.",
            "Stand up the public Hive gateway with mandatory signed manifests, sandboxing, revocation, reputation, and quotas.",
            "Use gas quotes and work receipts for all rented worker chunks before enabling any public market.",
            "Run MLX worker chunk smoke on the first Mac node that reports mlx availability.",
            "Promote heavier distributed training only after CUDA/MLX chunk placement and resource telemetry are stable.",
        ],
        confidence=0.72,
        market=market,
    )


def checkpoint_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    checkpoints = get_path(reports, ["checkpoint_registry", "checkpoints"], [])
    count = len(checkpoints) if isinstance(checkpoints, list) else 0
    latest = checkpoints[-1].get("checkpoint_id") if count and isinstance(checkpoints[-1], dict) else ""
    maturity = clamp(0.25 + min(count / 20.0, 1.0) * 0.45 + (0.15 if (ROOT / "scripts" / "checkpoint_chat.py").exists() else 0.0))
    return make_row(
        domain,
        status="active" if count else "partial",
        maturity=maturity,
        market_position="differentiated_research",
        evidence=[f"{count} checkpoints recorded", f"latest checkpoint {latest or 'none'}", "Major/minor materialization and checkpoint chat scripts exist."],
        gaps=["Minor-chain depth and restore smoke tests should be tracked over time.", "Model-weight checkpoint interaction is still state/report based, not full neural chat."],
        next_actions=["Run materialize/compare smoke after major changes.", "Expose checkpoint capability deltas in the matrix history."],
        confidence=0.79,
        market=market,
    )


def resource_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    resource = reports.get("resource_governor") if isinstance(reports.get("resource_governor"), dict) else {}
    performance = reports.get("performance_optimizer") if isinstance(reports.get("performance_optimizer"), dict) else {}
    gpu = get_path(resource, ["current_resources", "gpu"], {})
    gpu_available = bool(gpu.get("available"))
    efficiency = number(get_path(resource, ["efficiency", "score"], 0.0))
    can_run = get_path(resource, ["decision", "can_run_requested_profile"], None)
    owner = get_path(resource, ["decision", "execution_owner"], "")
    performance_score = number(performance.get("score", 0.0))
    maturity = clamp(
        0.2
        + efficiency * 0.3
        + performance_score * 0.2
        + (0.15 if gpu_available else 0.0)
        + (0.1 if owner == "rust_cuda" else 0.0)
        + (0.1 if can_run else 0.0)
    )
    return make_row(
        domain,
        status="ready" if gpu_available and can_run else "partial",
        maturity=maturity,
        market_position="local_specialization",
        evidence=[
            f"GPU {gpu.get('name', 'unknown')}",
            f"free VRAM {number(gpu.get('memory_free_mib', 0)):.0f} MiB",
            f"efficiency score {efficiency:.3f}",
            f"performance optimizer {performance.get('trigger_state', 'unknown')} score {performance_score:.3f}",
            f"training backend {get_path(performance, ['summary', 'preferred_training_backend'], owner or 'unknown')}",
            f"inference backend {get_path(performance, ['summary', 'preferred_inference_backend'], 'unknown')}",
        ],
        gaps=[
            "Longer candidate/seed-sweep VRAM stress evidence should stay current.",
            "More training hot loops should move from Python orchestration into Rust/CUDA or MLX worker chunks.",
        ],
        next_actions=[
            "Run VRAM stress before long profiles.",
            "Keep release builds and target-cpu native for CPU-heavy paths.",
            "Use performance_optimizer as the first bottleneck report before escalating architecture changes.",
        ],
        confidence=0.84,
        market=market,
    )


def language_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    ledger = reports.get("benchmark_ledger") if isinstance(reports.get("benchmark_ledger"), list) else []
    baby_rows = [row for row in ledger if "babylm" in str(row.get("benchmark_name", "")).lower()]
    frontier = next((row for row in baby_rows if row.get("lifecycle") == "frontier"), baby_rows[0] if baby_rows else {})
    score = number(frontier.get("score", 0.0))
    floor = number(get_path(frontier, ["graduation_policy", "floor_threshold"], 0.7))
    status = "active" if score >= floor else "partial"
    maturity = clamp(0.25 + min(len(baby_rows) / 2.0, 1.0) * 0.2 + score * 0.35 + (0.1 if frontier else 0.0))
    return make_row(
        domain,
        status=status,
        maturity=maturity,
        market_position="narrower_than_market",
        evidence=[f"{len(baby_rows)} BabyLM-related ledger entries", f"current BabyLM/mutated score {score:.3f}", f"floor {floor:.3f}", f"lifecycle {frontier.get('lifecycle', 'none')}"],
        gaps=["This is a narrow grammar/sequence track, not broad language-model competence.", "Seed55 split-quality warning should stay clearly labeled if used as a frontier."],
        next_actions=["Preserve public/local comparator.", "Generate harder leakage-checked mutated or live frontiers after graduation."],
        confidence=0.82,
        market=market,
    )


def rl_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    summary = get_path(reports, ["rl_registry", "summary"], {})
    local_envs = int(number(summary.get("local_envs", 0)))
    recs = get_path(reports, ["rl_registry", "recommended_frontier"], [])
    rec_count = len(recs) if isinstance(recs, list) else 0
    staged = int(number(get_path(reports, ["online_source_catalog", "summary", "imported_or_present"], 0)))
    maturity = clamp(0.2 + min(local_envs / 50.0, 1.0) * 0.25 + min(rec_count / 4.0, 1.0) * 0.2 + min(staged / 10.0, 1.0) * 0.15)
    return make_row(
        domain,
        status="partial" if rec_count else "planned",
        maturity=maturity,
        market_position="differentiated_research_but_early",
        evidence=[f"{local_envs} local RL envs", f"{rec_count} recommended RL frontiers", f"{staged} online sources staged/present"],
        gaps=["Many external RL sources are staged but not adapted into smoke-tested benchmark runners.", "Commercial ROMs or proprietary game assets remain blocked unless rights are explicit."],
        next_actions=["Add adapter smokes for the highest-value staged RL suites.", "Promote proven RL tasks into the benchmark ledger with regression gates."],
        confidence=0.68,
        market=market,
    )


def drone_racing_control_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    catalog_sources = get_path(reports, ["online_source_catalog", "sources"], [])
    drone_sources = [
        row
        for row in catalog_sources
        if isinstance(row, dict) and str(row.get("category", "")).startswith("drone_")
    ] if isinstance(catalog_sources, list) else []
    factory_cards = get_path(reports, ["benchmark_adapter_factory", "cards"], [])
    drone_cards = [
        row
        for row in factory_cards
        if isinstance(row, dict) and str(row.get("category", "")).startswith("drone_")
    ] if isinstance(factory_cards, list) else []
    smoke_passed = len([row for row in drone_cards if row.get("status") == "adapter_smoke_passed"])
    spec = reports.get("ai_grand_prix_spec") if isinstance(reports.get("ai_grand_prix_spec"), dict) else {}
    runtime = reports.get("python_runtime") if isinstance(reports.get("python_runtime"), dict) else {}
    runtime_ready = bool(get_path(runtime, ["summary", "ai_grand_prix_runtime_ready"], False))
    spec_ready = bool(get_path(spec, ["summary", "contract_recorded"], False))
    maturity = clamp(
        0.15
        + min(len(drone_sources) / 6.0, 1.0) * 0.2
        + min(len(drone_cards) / 6.0, 1.0) * 0.2
        + min(smoke_passed / 2.0, 1.0) * 0.15
        + (0.15 if spec_ready else 0.0)
        + (0.1 if runtime_ready else 0.0)
    )
    status = "ready" if smoke_passed and spec_ready and runtime_ready else ("active" if drone_sources and spec_ready else "partial")
    return make_row(
        domain,
        status=status,
        maturity=maturity,
        market_position="early_but_strategic",
        evidence=[
            f"{len(drone_sources)} governed drone source candidates",
            f"{len(drone_cards)} drone adapter cards",
            f"{smoke_passed} drone adapter smoke-passed cards",
            evidence_bool("AI Grand Prix spec contract recorded", spec_ready),
            evidence_bool("Python 3.14 runtime ready", runtime_ready),
        ],
        gaps=[
            "Competition-grade race simulation requires the simulator endpoint and Python 3.14 lane to be tested on the actual race stack.",
            "Live drone hardware must remain approval-gated with reflex/failsafe tests before any real flight.",
        ],
        next_actions=[
            "Smoke PyFlyt or gym-pybullet-drones first for hover/waypoint control.",
            "Create the .venv-drone-py314 runtime before official AI Grand Prix submissions.",
            "Add AirSim Drone Racing Lab endpoint health once the simulator is installed/running.",
        ],
        confidence=0.74,
        market=market,
    )


def minecraft_open_world_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    runtime = reports.get("minecraft_runtime_probe") if isinstance(reports.get("minecraft_runtime_probe"), dict) else {}
    runtime_summary = runtime.get("summary") if isinstance(runtime.get("summary"), dict) else {}
    factory_cards = get_path(reports, ["benchmark_adapter_factory", "cards"], [])
    minecraft_cards = [
        row
        for row in factory_cards
        if isinstance(row, dict) and str(row.get("category", "")) == "minecraft_rl_environment"
    ] if isinstance(factory_cards, list) else []
    smoke_passed = len([row for row in minecraft_cards if row.get("status") == "adapter_smoke_passed"])
    pressure_reports = list((ROOT / "reports").glob("pressure_source_crafter_seed*.json")) + list(
        (ROOT / "reports").glob("pressure_source_craftax_seed*.json")
    )
    full_ready = bool(runtime_summary.get("full_minecraft_runtime_ready"))
    bridge_ready = bool(runtime_summary.get("bridge_runtime_ready"))
    install_ready = bool(runtime_summary.get("local_minecraft_install_detected"))
    maturity = clamp(
        0.12
        + (0.16 if install_ready else 0.0)
        + (0.16 if bridge_ready else 0.0)
        + (0.18 if full_ready else 0.0)
        + min(len(minecraft_cards) / 5.0, 1.0) * 0.16
        + min(smoke_passed / 2.0, 1.0) * 0.14
        + min(len(pressure_reports) / 2.0, 1.0) * 0.12
    )
    status = "active" if pressure_reports else ("ready" if full_ready or bridge_ready or smoke_passed else "planned")
    return make_row(
        domain,
        status=status,
        maturity=maturity,
        market_position="strategic_open_world_frontier",
        evidence=[
            evidence_bool("local Minecraft install detected", install_ready),
            evidence_bool("bridge runtime ready", bridge_ready),
            evidence_bool("full Minecraft runtime ready", full_ready),
            f"{len(minecraft_cards)} Minecraft/Open-World adapter cards",
            f"{smoke_passed} Minecraft adapter smoke-passed cards",
            f"{len(pressure_reports)} Minecraft pressure reports",
        ],
        gaps=[
            "Full Minecraft harnesses still need runtime smoke before real long-horizon pressure.",
            "Player co-op and public/server modes remain intentionally gated.",
        ],
        next_actions=[
            "Run minecraft_runtime_probe each cycle.",
            "Smoke Crafter/Craftax bridge cards and then MineDojo/Malmo under the local license policy.",
            "Export trace capsules and transfer artifacts for navigation, inventory, survival, and crafting.",
        ],
        confidence=0.70,
        market=market,
    )


def native_voice_io_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    voice = reports.get("native_voice_io") if isinstance(reports.get("native_voice_io"), dict) else {}
    manifest = reports.get("native_voice_training_manifest") if isinstance(reports.get("native_voice_training_manifest"), dict) else {}
    score = number(get_path(voice, ["summary", "accuracy"], 0.0))
    cards = voice.get("voice_cards") if isinstance(voice.get("voice_cards"), list) else []
    learned_ready = bool(get_path(voice, ["summary", "native_model_ready"], False))
    owns_io = bool(get_path(voice, ["summary", "voice_is_head_router_io"], False))
    manifest_ready = bool(get_path(manifest, ["summary", "ready_for_native_training"], False))
    tiny_clips = int(number(get_path(manifest, ["summary", "tiny_audio_clips"], 0)))
    status = "frontier" if voice and not learned_ready else ("ready" if learned_ready else "planned")
    return make_row(
        domain,
        status=status,
        maturity=score if score else 0.18,
        market_position="native_local_frontier",
        evidence=[
            f"native voice score {score:.3f}",
            f"{len(cards)} voice benchmark/data cards tracked",
            evidence_bool("head/router owns voice I/O", owns_io),
            evidence_bool("licensed native voice training manifest ready", manifest_ready),
            f"{tiny_clips} tiny governed audio clips materialized",
            evidence_bool("native STT/TTS components ready", learned_ready),
            "Installed or provider STT/TTS inference is forbidden and does not count as progress.",
        ],
        gaps=[
            "Native STT and TTS learners are not trained yet.",
            "Large speech shards still need explicit storage/runtime plans before long native STT/TTS runs.",
            "WER/MOS-style metrics need to be backed by native Theseus reports, not package probes.",
        ],
        next_actions=[
            "Use the native voice training manifest as the automatic data entrypoint for LibriSpeech, LibriTTS, LJSpeech, Common Voice, and VCTK.",
            "Train native audio feature, STT, and TTS components inside the head/router I/O path.",
            "Keep external inference audit at zero for every non-teacher voice run.",
        ],
        confidence=0.72 if voice else 0.5,
        market=market,
    )


def training_data_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    summary = get_path(reports, ["training_data_inventory", "summary"], {})
    files = int(number(summary.get("files", 0)))
    bytes_total = int(number(summary.get("bytes", 0)))
    catalog = get_path(reports, ["online_source_catalog", "summary"], {})
    sources = int(number(catalog.get("sources", 0)))
    training_use_allowed = bool(catalog.get("training_use_allowed", False))
    maturity = clamp(0.25 + min(files / 500.0, 1.0) * 0.25 + min(sources / 10.0, 1.0) * 0.2 + (0.1 if not training_use_allowed else 0.05))
    return make_row(
        domain,
        status="active" if files else "partial",
        maturity=maturity,
        market_position="local_training_specific",
        evidence=[f"{files} inventory files", f"{bytes_total / (1024 * 1024):.1f} MiB tracked", f"{sources} catalog sources", evidence_bool("external training use currently allowed", training_use_allowed)],
        gaps=["External training use is intentionally blocked until provenance, license, sampling, and leakage gates pass.", "Bulk corpora are metadata-only until a sampling plan is approved."],
        next_actions=["Keep inventory fresh before each training run.", "Promote only audited sources into training profiles."],
        confidence=0.81,
        market=market,
    )


def synthetic_data_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    synthetic = reports.get("synthetic_data") if isinstance(reports.get("synthetic_data"), dict) else {}
    ready = bool(synthetic.get("training_ready"))
    accepted = int(number(synthetic.get("accepted_count", 0)))
    ratio = number(synthetic.get("blend_synthetic_ratio", 0.0))
    verification_ok = bool(get_path(synthetic, ["verification", "ok"], False))
    maturity = clamp(0.25 + (0.25 if ready else 0.0) + min(accepted / 1000.0, 1.0) * 0.2 + (0.15 if verification_ok else 0.0))
    return make_row(
        domain,
        status="active" if ready else "partial",
        maturity=maturity,
        market_position="differentiated_research",
        evidence=[evidence_bool("training ready", ready), f"{accepted} accepted synthetic pairs", f"blend ratio {ratio:.4f}", evidence_bool("verification ok", verification_ok)],
        gaps=["Need ongoing public/private delta checks to avoid synthetic overfitting.", "Teacher-generated data remains proposal-only unless approved."],
        next_actions=["Use residual-targeted synthetic data only within blend caps.", "Track effectiveness by residual cluster before/after."],
        confidence=0.8,
        market=market,
    )


def context_packet_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    summary = get_path(reports, ["context_packets", "summary"], {})
    vcm = reports.get("virtual_context_memory") if isinstance(reports.get("virtual_context_memory"), dict) else {}
    vcm_summary = vcm.get("summary") if isinstance(vcm.get("summary"), dict) else {}
    vcm_bench = reports.get("virtual_context_memory_bench") if isinstance(reports.get("virtual_context_memory_bench"), dict) else {}
    vcm_index = reports.get("virtual_context_memory_index") if isinstance(reports.get("virtual_context_memory_index"), dict) else {}
    vcm_status = reports.get("virtual_context_memory_status") if isinstance(reports.get("virtual_context_memory_status"), dict) else {}
    vcm_training = reports.get("virtual_context_memory_training_admission") if isinstance(reports.get("virtual_context_memory_training_admission"), dict) else {}
    vcm_consumer_audit = reports.get("virtual_context_memory_consumer_audit") if isinstance(reports.get("virtual_context_memory_consumer_audit"), dict) else {}
    vcm_context_recovery = reports.get("vcm_context_recovery_benchmark") if isinstance(reports.get("vcm_context_recovery_benchmark"), dict) else {}
    vcm_green = vcm.get("trigger_state") == "GREEN"
    bench_green = vcm_bench.get("trigger_state") == "GREEN"
    bench_v2 = vcm_bench.get("policy") == "project_theseus_vcm_bench_v2"
    training_green = vcm_training.get("trigger_state") == "GREEN"
    consumer_audit_green = vcm_consumer_audit.get("trigger_state") == "GREEN"
    context_recovery_green = vcm_context_recovery.get("trigger_state") == "GREEN"
    packets = int(number(summary.get("packet_count", 0)))
    active = int(number(summary.get("active_packet_count", 0)))
    summaries = int(number(summary.get("summary_packet_count", 0)))
    top_score = number(summary.get("top_score", 0.0))
    vcm_pages = int(number(vcm_summary.get("semantic_pages", 0)))
    vcm_events = int(number(vcm_summary.get("event_count", 0)))
    vcm_edges = int(number(vcm_summary.get("graph_edge_count", 0)))
    vcm_index_pages = int(number(vcm_index.get("page_count", 0)))
    vcm_usage_pages = int(number(vcm_summary.get("usage_event_pages", 0)))
    context_recovery_accuracy = number(get_path(vcm_context_recovery, ["summary", "vcm_answer_accuracy"], 0.0))
    context_recovery_baseline = number(get_path(vcm_context_recovery, ["summary", "best_baseline_answer_accuracy"], 0.0))
    maturity = clamp(
        0.20
        + min(active / 64.0, 1.0) * 0.15
        + min(summaries / 10.0, 1.0) * 0.10
        + min(top_score / 4.0, 1.0) * 0.10
        + (0.15 if vcm_green else 0.0)
        + (0.15 if bench_green else 0.0)
        + min(vcm_pages / 128.0, 1.0) * 0.08
        + min(vcm_events / 32.0, 1.0) * 0.04
        + min(vcm_edges / 256.0, 1.0) * 0.03
        + (0.05 if bench_v2 else 0.0)
        + (0.05 if training_green else 0.0)
        + (0.04 if consumer_audit_green else 0.0)
        + min(vcm_index_pages / max(1.0, float(vcm_pages or 1)), 1.0) * 0.04
        + min(vcm_usage_pages / 4.0, 1.0) * 0.02
        + (0.04 if context_recovery_green else 0.0)
        + max(0.0, context_recovery_accuracy - context_recovery_baseline) * 0.04
    )
    return make_row(
        domain,
        status="active" if packets and vcm_green else "partial",
        maturity=maturity,
        market_position="differentiated_research",
        evidence=[
            f"{packets} packets",
            f"{active} active packets",
            f"{summaries} summary packets",
            f"top importance score {top_score:.3f}",
            f"VCM state {vcm.get('trigger_state', 'missing')} pages={vcm_pages} events={vcm_events} edges={vcm_edges}",
            f"VCM-Bench {vcm_bench.get('trigger_state', 'missing')} policy={vcm_bench.get('policy', 'missing')}",
            f"VCM query index pages={vcm_index_pages}",
            f"VCM training admission {vcm_training.get('trigger_state', 'missing')}",
            f"VCM consumer audit {vcm_consumer_audit.get('trigger_state', 'missing')} packet_only={get_path(vcm_consumer_audit, ['summary', 'packet_only_consumer_count'], 'missing')}",
            f"VCM status faults={get_path(vcm_status, ['summary', 'fault_count'], 'missing')} conflicts={get_path(vcm_status, ['summary', 'conflict_edge_count'], 'missing')}",
            f"VCM context recovery {vcm_context_recovery.get('trigger_state', 'missing')} accuracy={context_recovery_accuracy:.3f} best_baseline={context_recovery_baseline:.3f}",
        ],
        gaps=["Native runtime/KV-cache VCM remains future work.", "Need more real dogfood traces to test long-horizon memory under daily-use pressure.", "Public long-context benchmarks remain calibration-only until an explicit governed run."],
        next_actions=["Keep VCM refresh and context-recovery benchmark in every autonomy cycle.", "Use VCM query/explain for memory debugging.", "Route memory-derived training through the VCM admission bridge."],
        confidence=0.84,
        market=market,
    )


def safety_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    arms = get_path(reports, ["arm_registry", "arms"], [])
    safety_arm = any("safety" in str(row.get("arm_name", "")).lower() for row in arms) if isinstance(arms, list) else False
    safety_ledger = bool(reports.get("safety_ledger"))
    launch_ready = bool(get_path(reports, ["autonomy_launch_readiness", "ready_for_autonomous_training"], False))
    maturity = clamp(0.25 + (0.2 if safety_arm else 0.0) + (0.2 if safety_ledger else 0.0) + (0.15 if launch_ready else 0.0))
    return make_row(
        domain,
        status="active" if safety_arm and launch_ready else "partial",
        maturity=maturity,
        market_position="parity_for_local_research",
        evidence=[evidence_bool("safety/reflex arm present", safety_arm), evidence_bool("safety ledger present", safety_ledger), evidence_bool("launch readiness green", launch_ready), "Network fetch, teacher writes, promotions, and destructive actions are gated."],
        gaps=["This is a research safety layer, not certified safety-critical runtime verification.", "Need stronger automated permission-audit tests for new arms/tools."],
        next_actions=["Run safety/permission checks whenever arms or tools change.", "Keep high-risk actions human-approved."],
        confidence=0.79,
        market=market,
    )


def attd_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    attd = reports.get("attd") if isinstance(reports.get("attd"), dict) else {}
    packets = reports.get("attd_packets") if isinstance(reports.get("attd_packets"), dict) else {}
    state = str(attd.get("trigger_state") or "missing")
    score = number(attd.get("attd_score", 1.0 if not attd else 0.0))
    components = attd.get("components") if isinstance(attd.get("components"), dict) else {}
    packet_count = int(number(packets.get("packet_count", 0)))
    status = "blocked" if state == "RED" else ("active" if state == "YELLOW" else ("ready" if state == "GREEN" else "partial"))
    maturity = clamp(0.25 + (0.35 if attd else 0.0) + (0.2 if state in {"GREEN", "YELLOW"} else 0.0) + (0.1 if packet_count or state == "GREEN" else 0.0) + max(0.0, 0.1 - score * 0.1))
    return make_row(
        domain,
        status=status,
        maturity=maturity,
        market_position="differentiated_research",
        evidence=[
            f"ATTD trigger state {state}",
            f"ATTD score {score:.3f}",
            f"{packet_count} maintenance packets",
            f"components {', '.join(f'{k}={number(v):.2f}' for k, v in list(components.items())[:5])}",
        ],
        gaps=[
            "ATTD metrics are deterministic proxies and should be calibrated against longer repository history.",
            "Maintenance packets are generated but still need measured before/after validation as the system evolves.",
        ],
        next_actions=[
            "Run ATTD every autonomy cycle before teacher self-edit, architecture change, adapter card writes, and launch readiness.",
            "Use RED only to force maintenance, not to hide useful simplification work.",
        ],
        confidence=0.8 if attd else 0.45,
        market=market,
    )


def codebase_engineering_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    goal = reports.get("autonomous_goal") if isinstance(reports.get("autonomous_goal"), dict) else {}
    traces = ROOT / "reports" / "workflow_routing_traces.jsonl"
    trace_count = jsonl_line_count(traces, max_lines=10000)
    selected = get_path(goal, ["selected_arms"], [])
    maturity = clamp(0.2 + min(trace_count / 100.0, 1.0) * 0.25 + (0.15 if selected else 0.0))
    return make_row(
        domain,
        status="partial",
        maturity=maturity,
        market_position="behind_market_products",
        evidence=[f"{trace_count} workflow routing traces", f"latest selected arms {', '.join(selected) if isinstance(selected, list) else selected}", "Local system can route goals and run bounded commands, but PR-grade software engineering still relies on Codex/teacher supervision."],
        gaps=["No fully independent arbitrary-repo code-edit/PR loop owned by the local model.", "Market agents are much stronger for general codebase engineering today."],
        next_actions=["Keep Codex/teacher as supervised bootstrapping layer.", "Use codebase tasks as future benchmark/regression surfaces once local agent ownership improves."],
        confidence=0.74,
        market=market,
    )


def external_research_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    seeker = reports.get("benchmark_seeker") if isinstance(reports.get("benchmark_seeker"), dict) else {}
    discovered = get_path(seeker, ["discovered_external_candidates"], [])
    queued = get_path(seeker, ["queued_external_candidates"], [])
    knowledge = get_path(reports, ["knowledge_sources", "sources"], [])
    catalog_sources = int(number(get_path(reports, ["online_source_catalog", "summary", "sources"], 0)))
    maturity = clamp(0.25 + min(catalog_sources / 10.0, 1.0) * 0.25 + min(len(discovered) / 10.0, 1.0) * 0.15 + min(len(knowledge) / 3.0, 1.0) * 0.1)
    return make_row(
        domain,
        status="active" if catalog_sources else "partial",
        maturity=maturity,
        market_position="more_governed_less_capable_than_web_agents",
        evidence=[f"{catalog_sources} curated online catalog sources", f"{len(discovered) if isinstance(discovered, list) else 0} discovered external candidates", f"{len(queued) if isinstance(queued, list) else 0} queued candidates", f"{len(knowledge) if isinstance(knowledge, list) else 0} knowledge sources"],
        gaps=["Network fetch is intentionally off by default.", "Lookup-only sources need terms, robots, and license audit before training use."],
        next_actions=["Audit and adapter-smoke staged sources.", "Keep Grokipedia and similar sources lookup-only until terms/provenance gates clear."],
        confidence=0.75,
        market=market,
    )


def capability_matrix_row(reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    config_exists = (ROOT / "configs" / "capability_market_baselines.json").exists()
    return make_row(
        {
            "id": "live_capability_matrix",
            "name": "Live capability matrix",
            "category": "meta_governance",
            "target": "Continuously expose current local capabilities, gaps, next actions, and market posture.",
            "market_comparators": ["openai_codex", "anthropic_claude_code", "google_gemini_cli", "github_copilot_cloud_agent", "cursor", "devin"],
            "market_standard": "Products expose feature pages and dashboards; SparkStream now exposes a machine-readable self-audit tied to local reports.",
            "local_evidence_paths": ["reports/capability_matrix.json", "reports/capability_matrix.md", "configs/capability_market_baselines.json"],
        },
        status="ready" if config_exists else "partial",
        maturity=0.86 if config_exists else 0.55,
        market_position="new_local_advantage",
        evidence=[evidence_bool("baseline config present", config_exists), "current matrix report is generated by this run", "Matrix is wired into autonomy cycle and dashboard by this change."],
        gaps=["Market baselines still need periodic source refresh.", "Capability trends need multiple matrix snapshots."],
        next_actions=["Regenerate every autonomy cycle.", "Add trend charts after several cycles."],
        confidence=0.76,
        market=market,
    )


def generic_row(domain: dict[str, Any], reports: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    evidence_paths = domain.get("local_evidence_paths", [])
    present = [path for path in evidence_paths if (ROOT / path).exists()]
    maturity = clamp(0.25 + (len(present) / max(1, len(evidence_paths))) * 0.5)
    status = "active" if present else "planned"
    return make_row(
        domain,
        status=status,
        maturity=maturity,
        market_position="unclassified",
        evidence=[f"{len(present)}/{len(evidence_paths)} evidence paths present"],
        gaps=["Add a typed row builder for this domain if it becomes important."],
        next_actions=["Refresh reports and classify this domain."],
        confidence=0.5,
        market=market,
    )


def make_row(
    domain: dict[str, Any],
    *,
    status: str,
    maturity: float,
    market_position: str,
    evidence: list[str],
    gaps: list[str],
    next_actions: list[str],
    confidence: float,
    market: dict[str, Any],
) -> dict[str, Any]:
    comparators = []
    for system_id in domain.get("market_comparators", []):
        system = market.get(system_id, {})
        if system:
            comparators.append(
                {
                    "id": system.get("id"),
                    "name": system.get("name"),
                    "source_url": system.get("source_url"),
                    "tags": system.get("comparison_tags", []),
                    "notes": system.get("capability_notes", []),
                }
            )
    evidence_paths = []
    for path in domain.get("local_evidence_paths", []):
        local = ROOT / path
        evidence_paths.append(
            {
                "path": str(path).replace("\\", "/"),
                "exists": local.exists(),
            }
        )
    return {
        "capability_id": domain.get("id"),
        "name": domain.get("name"),
        "category": domain.get("category"),
        "target": domain.get("target"),
        "status": status,
        "maturity": round(clamp(maturity), 3),
        "confidence": round(clamp(confidence), 3),
        "market_position": market_position,
        "market_standard": domain.get("market_standard"),
        "market_comparators": comparators,
        "local_evidence": evidence,
        "local_evidence_paths": evidence_paths,
        "gaps": gaps,
        "next_actions": next_actions,
    }


def build_summary(config: dict[str, Any], reports: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    maturity_values = [number(row.get("maturity", 0.0)) for row in rows]
    ledger = reports.get("benchmark_ledger") if isinstance(reports.get("benchmark_ledger"), list) else []
    arms = get_path(reports, ["arm_registry", "arms"], [])
    tools = get_path(reports, ["tool_registry", "tools"], [])
    frontiers = [row for row in ledger if isinstance(row, dict) and row.get("lifecycle") == "frontier"]
    regressions = [row for row in ledger if isinstance(row, dict) and row.get("lifecycle") == "regression"]
    behind = [row for row in rows if "behind" in str(row.get("market_position", ""))]
    differentiated = [row for row in rows if "differentiated" in str(row.get("market_position", ""))]
    top_gaps: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: number(item.get("maturity", 0.0))):
        for gap in row.get("gaps", [])[:1]:
            top_gaps.append({"capability_id": row.get("capability_id", ""), "gap": gap})
        if len(top_gaps) >= 8:
            break
    return {
        "capabilities": len(rows),
        "statuses": statuses,
        "average_maturity": round(sum(maturity_values) / max(1, len(maturity_values)), 3),
        "ready_or_active": statuses.get("ready", 0) + statuses.get("active", 0),
        "partial_or_blocked": statuses.get("partial", 0) + statuses.get("blocked", 0) + statuses.get("planned", 0),
        "market_systems": len(config.get("market_systems", [])),
        "behind_market_count": len(behind),
        "differentiated_count": len(differentiated),
        "benchmark_frontiers": len(frontiers),
        "benchmark_regressions": len(regressions),
        "arms": len(arms) if isinstance(arms, list) else 0,
        "tools": len(tools) if isinstance(tools, list) else 0,
        "launch_ready": bool(get_path(reports, ["autonomy_launch_readiness", "ready_for_autonomous_training"], False)),
        "teacher_ready": bool(get_path(reports, ["autonomy_launch_readiness", "ready_for_teacher_enabled_run"], False)),
        "attd_trigger_state": get_path(reports, ["attd", "trigger_state"], ""),
        "attd_score": get_path(reports, ["attd", "attd_score"], None),
        "legacy_port_mechanisms": get_path(reports, ["legacy_port_mechanisms", "summary", "mechanisms"], None),
        "legacy_port_red": get_path(reports, ["legacy_port_mechanisms", "summary", "red"], None),
        "legacy_port_yellow_or_degraded": get_path(reports, ["legacy_port_mechanisms", "summary", "yellow_or_degraded"], None),
        "gpu": get_path(reports, ["resource_governor", "current_resources", "gpu", "name"], ""),
        "hive_nodes": get_path(reports, ["hive_scheduler", "summary", "nodes"], None),
        "hive_peers": get_path(reports, ["hive_peers", "peer_count"], None),
        "top_gaps": top_gaps,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SparkStream Capability Matrix",
        "",
        f"Generated: `{payload.get('created_utc')}`",
        "",
        "This is generated from local reports plus source-backed market baseline configuration.",
        "",
        "## Summary",
        "",
    ]
    summary = payload.get("summary", {})
    lines.extend(
        [
            f"- Capabilities: `{summary.get('capabilities')}`",
            f"- Average maturity: `{summary.get('average_maturity')}`",
            f"- Ready/active: `{summary.get('ready_or_active')}`",
            f"- Partial/planned/blocked: `{summary.get('partial_or_blocked')}`",
            f"- Market systems compared: `{summary.get('market_systems')}`",
            f"- Differentiated domains: `{summary.get('differentiated_count')}`",
            f"- Behind-market domains: `{summary.get('behind_market_count')}`",
            "",
            "## Matrix",
            "",
            "| Capability | Status | Maturity | Market Position | Key Gap |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for row in payload.get("matrix", []):
        gap = (row.get("gaps") or [""])[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(row.get("name", "")),
                    escape_md(row.get("status", "")),
                    f"{number(row.get('maturity', 0.0)):.3f}",
                    escape_md(row.get("market_position", "")),
                    escape_md(gap),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Market Sources", ""])
    for system in payload.get("market_systems", []):
        lines.append(f"- [{system.get('name')}]({system.get('source_url')})")
    lines.append("")
    return "\n".join(lines)


def source_reports() -> list[dict[str, Any]]:
    paths = [
        "reports/architecture_gate_report.json",
        "reports/attd_report.json",
        "reports/attd_maintenance_packets.json",
        "reports/arm_lifecycle_governance.json",
        "reports/arm_registry.json",
        "reports/arm_sucker_registry.json",
        "reports/arm_sucker_registry.md",
        "reports/ai_grand_prix_spec_digest.json",
        "reports/minecraft_runtime_probe.json",
        "reports/autonomy_launch_readiness.json",
        "reports/benchmark_ledger.json",
        "reports/candidate_promotion_gate.json",
        "reports/candidate_bottleneck_reducer.json",
        "reports/checkpoint_registry.json",
        "reports/context_packet_ledger.json",
        "reports/hive_status.json",
        "reports/hive_peers.json",
        "reports/hive_scheduler.json",
        "reports/hive_relay_status.json",
        "reports/license_status.json",
        "reports/compute_market_status.json",
        "reports/public_hive_contribution_status.json",
        "reports/online_source_catalog_report.json",
        "reports/openai_compat_status.json",
        "reports/update_status.json",
        "reports/update_offer_current.json",
        "reports/public_comparator_ledger.json",
        "reports/resource_governor.json",
        "reports/performance_optimizer.json",
        "reports/legacy_port_mechanisms.json",
        "reports/planforge_schedule.json",
        "reports/coherence_delirium_report.json",
        "reports/proxy_truth_audit.json",
        "reports/world_adapter_job_runtime.json",
        "reports/emulator_game_trace_gateway.json",
        "reports/active_inference_world_model.json",
        "reports/macro_counterexample_gate.json",
        "reports/bridge_adapter_native_promotion.json",
        "reports/pretraining_readiness_integrity.json",
        "reports/salience_scheduler.json",
        "reports/campaign_dag.json",
        "reports/dataset_recipe_scaffolder.json",
        "reports/evidence_graph_ledger.json",
        "reports/runtime_resolution_boundary.json",
        "reports/tiered_memory_consolidation.json",
        "reports/aletheia_advocate_gate.json",
        "reports/synaptic_work_stealing.json",
        "reports/architecture_motif_library.json",
        "reports/semantic_intent_repair.json",
        "reports/eval_track_contract_library.json",
        "reports/synaptic_permission_decay.json",
        "reports/temporal_replay_assertions.json",
        "reports/whitecell_threat_memory.json",
        "reports/zero_copy_context_prefetch.json",
        "reports/hil_emulator_gate.json",
        "reports/formal_runtime_coupling.json",
        "reports/veritas_discovery_novelty.json",
        "reports/anti_expert_tribunal_router.json",
        "reports/probe_router_burst_budget.json",
        "reports/rlds_minari_trace_export.json",
        "reports/live_operator_advisors.json",
        "reports/benchmark_bounty_registry.json",
        "reports/legacy_fine_tooth_comb.json",
        "reports/python_runtime_compatibility.json",
        "reports/rl_benchmark_registry.json",
        "reports/ratcheting_modular_intelligence_report.json",
        "reports/safety_benchmark_ledger.json",
        "reports/synthetic_data_curator.json",
        "reports/tool_registry.json",
        "reports/training_data_inventory.json",
    ]
    rows = []
    for path in paths:
        local = ROOT / path
        rows.append(
            {
                "path": path,
                "exists": local.exists(),
                "bytes": local.stat().st_size if local.exists() else 0,
            }
        )
    return rows


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def jsonl_line_count(path: Path, *, max_lines: int) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for count, _line in enumerate(handle, start=1):
            if count >= max_lines:
                return count
    return count


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def evidence_bool(label: str, value: Any) -> str:
    if value is None:
        return f"{label}: unknown"
    return f"{label}: {'yes' if bool(value) else 'no'}"


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
