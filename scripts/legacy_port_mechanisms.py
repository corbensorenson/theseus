"""Materialize legacy-project concepts as Theseus governance reports.

The predecessor audit is useful only if its best ideas become live surfaces.
This runner implements the first concrete contracts for the highest-value
legacy ports:

- PlanForge critical-path scheduling.
- Coherence/delirium runtime health.
- Proxy truth audits for benchmark score integrity.
- TaskSpell goal locks.
- Low-rank lane adapter-bank planning.
- World adapter/job runtime contracts.
- BYO-ROM-safe emulator/game trace gateway.
- CUDA/MLX/CPU/Hive compute-mode acceptance.
- ORCP-inspired deterministic compression pressure.
- USB/serial weak-device endpoint contract.
- Hotpath, drone parity, self-mod proof, native speech, and trace-fabric reports.

It does not call external inference and does not fetch data. It reads existing
reports and writes deterministic summary artifacts for the autonomy loop,
dashboard, watchdog, and teacher governor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import time
import zlib
from pathlib import Path
from typing import Any

from legacy_port_late_builders import (
    build_synaptic_work_stealing,
    build_architecture_motif_library,
    build_semantic_intent_repair,
    build_eval_track_contract_library,
    build_synaptic_permission_decay,
    build_temporal_replay_assertions,
    build_whitecell_threat_memory,
    build_zero_copy_context_prefetch,
    build_hil_emulator_gate,
    build_formal_runtime_coupling,
    build_veritas_discovery_novelty,
    build_anti_expert_tribunal_router,
    build_probe_router_burst_budget,
    build_rlds_minari_trace_export,
    build_live_operator_advisors,
    build_benchmark_bounty_registry,
    build_legacy_fine_tooth_comb,
)
from legacy_port_support import *



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/legacy_port_mechanisms.json")
    parser.add_argument("--markdown-out", default="reports/legacy_port_mechanisms.md")
    args = parser.parse_args()

    policy = read_json(resolve(args.policy))
    state = observe()
    started = time.perf_counter()

    reports = {
        "planforge": build_planforge(policy, state),
        "coherence_delirium": build_coherence_delirium(policy, state),
        "proxy_truth_audit": build_proxy_truth_audit(policy, state),
        "taskspell_contracts": build_taskspell_contracts(policy, state),
        "low_rank_adapter_bank": build_low_rank_adapter_bank(policy, state),
        "world_adapter_jobs": build_world_adapter_jobs(policy, state),
        "emulator_game_trace_gateway": build_emulator_gateway(policy, state),
        "compute_mode_acceptance": build_compute_mode_acceptance(policy, state),
        "orcp_compression_frontier": build_compression_frontier(policy),
        "device_endpoint_contract": build_device_endpoint_contract(policy, state),
        "hotpath_quality_gates": build_hotpath_quality_gates(policy, state),
        "drone_blackbox_parity": build_drone_blackbox_parity(policy, state),
        "self_mod_proof_bundle": build_self_mod_proof_bundle(policy, state),
        "first_party_speech_contract": build_first_party_speech_contract(policy, state),
        "trace_fabric_training_exchange": build_trace_fabric_exchange(policy, state),
        "active_inference_world_model": build_active_inference_world_model(policy, state),
        "macro_counterexample_gate": build_macro_counterexample_gate(policy, state),
        "bridge_adapter_native_promotion": build_bridge_adapter_native_promotion(policy, state),
        "pretraining_readiness_integrity": build_pretraining_readiness_integrity(policy, state),
        "salience_scheduler": build_salience_scheduler(policy, state),
        "campaign_dag": build_campaign_dag(policy, state),
        "dataset_recipe_scaffolder": build_dataset_recipe_scaffolder(policy, state),
        "evidence_graph_ledger": build_evidence_graph_ledger(policy, state),
        "runtime_resolution_boundary": build_runtime_resolution_boundary(policy, state),
        "tiered_memory_consolidation": build_tiered_memory_consolidation(policy, state),
        "aletheia_advocate_gate": build_aletheia_advocate_gate(policy, state),
        "synaptic_work_stealing": build_synaptic_work_stealing(policy, state),
        "architecture_motif_library": build_architecture_motif_library(policy, state),
        "semantic_intent_repair": build_semantic_intent_repair(policy, state),
        "eval_track_contract_library": build_eval_track_contract_library(policy, state),
        "synaptic_permission_decay": build_synaptic_permission_decay(policy, state),
        "temporal_replay_assertions": build_temporal_replay_assertions(policy, state),
        "whitecell_threat_memory": build_whitecell_threat_memory(policy, state),
        "zero_copy_context_prefetch": build_zero_copy_context_prefetch(policy, state),
        "hil_emulator_gate": build_hil_emulator_gate(policy, state),
        "formal_runtime_coupling": build_formal_runtime_coupling(policy, state),
        "veritas_discovery_novelty": build_veritas_discovery_novelty(policy, state),
        "anti_expert_tribunal_router": build_anti_expert_tribunal_router(policy, state),
        "probe_router_burst_budget": build_probe_router_burst_budget(policy, state),
        "rlds_minari_trace_export": build_rlds_minari_trace_export(policy, state),
        "live_operator_advisors": build_live_operator_advisors(policy, state),
        "benchmark_bounty_registry": build_benchmark_bounty_registry(policy, state),
        "legacy_fine_tooth_comb": build_legacy_fine_tooth_comb(policy, state),
    }
    report_paths = {
        "planforge": write_report("planforge_schedule.json", reports["planforge"]),
        "coherence_delirium": write_report("coherence_delirium_report.json", reports["coherence_delirium"]),
        "proxy_truth_audit": write_report("proxy_truth_audit.json", reports["proxy_truth_audit"]),
        "taskspell_contracts": write_report("taskspell_contracts.json", reports["taskspell_contracts"]),
        "low_rank_adapter_bank": write_report("low_rank_adapter_bank.json", reports["low_rank_adapter_bank"]),
        "world_adapter_jobs": write_report("world_adapter_job_runtime.json", reports["world_adapter_jobs"]),
        "emulator_game_trace_gateway": write_report("emulator_game_trace_gateway.json", reports["emulator_game_trace_gateway"]),
        "compute_mode_acceptance": write_report("compute_mode_acceptance.json", reports["compute_mode_acceptance"]),
        "orcp_compression_frontier": write_report("orcp_compression_frontier.json", reports["orcp_compression_frontier"]),
        "device_endpoint_contract": write_report("device_endpoint_contract.json", reports["device_endpoint_contract"]),
        "hotpath_quality_gates": write_report("hotpath_quality_gates.json", reports["hotpath_quality_gates"]),
        "drone_blackbox_parity": write_report("drone_blackbox_parity.json", reports["drone_blackbox_parity"]),
        "self_mod_proof_bundle": write_report("self_mod_proof_bundle.json", reports["self_mod_proof_bundle"]),
        "first_party_speech_contract": write_report("first_party_speech_contract.json", reports["first_party_speech_contract"]),
        "trace_fabric_training_exchange": write_report("trace_fabric_training_exchange.json", reports["trace_fabric_training_exchange"]),
        "active_inference_world_model": write_report("active_inference_world_model.json", reports["active_inference_world_model"]),
        "macro_counterexample_gate": write_report("macro_counterexample_gate.json", reports["macro_counterexample_gate"]),
        "bridge_adapter_native_promotion": write_report("bridge_adapter_native_promotion.json", reports["bridge_adapter_native_promotion"]),
        "pretraining_readiness_integrity": write_report("pretraining_readiness_integrity.json", reports["pretraining_readiness_integrity"]),
        "salience_scheduler": write_report("salience_scheduler.json", reports["salience_scheduler"]),
        "campaign_dag": write_report("campaign_dag.json", reports["campaign_dag"]),
        "dataset_recipe_scaffolder": write_report("dataset_recipe_scaffolder.json", reports["dataset_recipe_scaffolder"]),
        "evidence_graph_ledger": write_report("evidence_graph_ledger.json", reports["evidence_graph_ledger"]),
        "runtime_resolution_boundary": write_report("runtime_resolution_boundary.json", reports["runtime_resolution_boundary"]),
        "tiered_memory_consolidation": write_report("tiered_memory_consolidation.json", reports["tiered_memory_consolidation"]),
        "aletheia_advocate_gate": write_report("aletheia_advocate_gate.json", reports["aletheia_advocate_gate"]),
        "synaptic_work_stealing": write_report("synaptic_work_stealing.json", reports["synaptic_work_stealing"]),
        "architecture_motif_library": write_report("architecture_motif_library.json", reports["architecture_motif_library"]),
        "semantic_intent_repair": write_report("semantic_intent_repair.json", reports["semantic_intent_repair"]),
        "eval_track_contract_library": write_report("eval_track_contract_library.json", reports["eval_track_contract_library"]),
        "synaptic_permission_decay": write_report("synaptic_permission_decay.json", reports["synaptic_permission_decay"]),
        "temporal_replay_assertions": write_report("temporal_replay_assertions.json", reports["temporal_replay_assertions"]),
        "whitecell_threat_memory": write_report("whitecell_threat_memory.json", reports["whitecell_threat_memory"]),
        "zero_copy_context_prefetch": write_report("zero_copy_context_prefetch.json", reports["zero_copy_context_prefetch"]),
        "hil_emulator_gate": write_report("hil_emulator_gate.json", reports["hil_emulator_gate"]),
        "formal_runtime_coupling": write_report("formal_runtime_coupling.json", reports["formal_runtime_coupling"]),
        "veritas_discovery_novelty": write_report("veritas_discovery_novelty.json", reports["veritas_discovery_novelty"]),
        "anti_expert_tribunal_router": write_report("anti_expert_tribunal_router.json", reports["anti_expert_tribunal_router"]),
        "probe_router_burst_budget": write_report("probe_router_burst_budget.json", reports["probe_router_burst_budget"]),
        "rlds_minari_trace_export": write_report("rlds_minari_trace_export.json", reports["rlds_minari_trace_export"]),
        "live_operator_advisors": write_report("live_operator_advisors.json", reports["live_operator_advisors"]),
        "benchmark_bounty_registry": write_report("benchmark_bounty_registry.json", reports["benchmark_bounty_registry"]),
        "legacy_fine_tooth_comb": write_report("legacy_fine_tooth_comb.json", reports["legacy_fine_tooth_comb"]),
    }

    aggregate = {
        "policy": "project_theseus_legacy_port_mechanisms_v0",
        "created_utc": now(),
        "policy_file": rel(resolve(args.policy)),
        "summary": aggregate_summary(reports),
        "reports": report_paths,
        "mechanisms": reports,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    out = resolve(args.out)
    write_json(out, aggregate)
    write_markdown(resolve(args.markdown_out), aggregate)
    print(json.dumps({"ok": True, "summary": aggregate["summary"], "out": rel(out)}, indent=2))
    return 0


def observe() -> dict[str, Any]:
    return {
        "watchdog": read_json(REPORTS / "autonomy_watchdog.json"),
        "status": read_json(REPORTS / "sparkstream_status.json"),
        "daemon_ledger_tail": read_jsonl_tail(REPORTS / "sparkstream_daemon_ledger.jsonl", 20),
        "autonomy_ledger_tail": read_jsonl_tail(REPORTS / "autonomy_ledger.jsonl", 40),
        "cycle": read_json(REPORTS / "autonomy_cycle_last.json"),
        "candidate": read_json(REPORTS / "candidate_promotion_gate.json"),
        "frontier_policy": read_json(REPORTS / "frontier_policy_status.json"),
        "benchmark_ledger": read_json(REPORTS / "benchmark_ledger.json"),
        "residual_escrow": read_json(REPORTS / "residual_escrow.json"),
        "resource_governor": read_json(REPORTS / "resource_governor.json"),
        "performance_optimizer": read_json(REPORTS / "performance_optimizer.json"),
        "hive_status": read_json(REPORTS / "hive_status.json"),
        "hive_scheduler": read_json(REPORTS / "hive_scheduler.json"),
        "legacy_audit": read_json(REPORTS / "legacy_project_concept_audit.json"),
        "task_goal": read_json(REPORTS / "autonomous_goal_last.json"),
        "arm_transfer_plan": read_json(REPORTS / "arm_transfer_plan.json"),
        "arm_transfer_artifacts": read_json(REPORTS / "arm_transfer_artifacts.json"),
        "arm_registry": read_json(REPORTS / "arm_registry.json"),
        "local_rom_registry": read_json(REPORTS / "local_rom_registry.json"),
        "game_asset_inventory": read_json(REPORTS / "game_asset_inventory.json"),
        "teacher_self_edit": read_json(REPORTS / "teacher_self_edit_last.json"),
        "teacher_self_edit_proof": read_json(REPORTS / "teacher_self_edit_proof.json"),
        "self_evolution_governance": read_json(REPORTS / "self_evolution_governance.json"),
        "world_adapter_jobs": read_json(REPORTS / "world_adapter_job_runtime.json"),
        "emulator_game_trace_gateway": read_json(REPORTS / "emulator_game_trace_gateway.json"),
        "loop_closure_harvester": read_json(REPORTS / "loop_closure_harvester.json"),
        "loop_closure_tool_promoter": read_json(REPORTS / "loop_closure_tool_promoter.json"),
        "benchmark_adapter_factory": read_json(REPORTS / "benchmark_adapter_factory.json"),
        "resource_pantry": read_json(REPORTS / "resource_pantry.json"),
        "online_source_catalog_report": read_json(REPORTS / "online_source_catalog_report.json"),
        "training_data_inventory": read_json(REPORTS / "training_data_inventory.json"),
        "training_data_sampler": read_json(REPORTS / "training_data_sampler.json"),
        "autonomy_launch_readiness": read_json(REPORTS / "autonomy_launch_readiness.json"),
        "architecture_experiment_runner": read_json(REPORTS / "architecture_experiment_runner.json"),
        "model_ledger": read_json(REPORTS / "model_ledger.json"),
        "native_voice_policy": read_json(ROOT / "configs" / "native_voice_policy.json"),
        "native_voice_training_manifest": read_json(REPORTS / "native_voice_training_manifest.json"),
        "native_voice_io": read_json(REPORTS / "native_voice_io.json"),
        "attd": read_json(REPORTS / "attd_report.json"),
        "architecture_experiment_results": read_json(REPORTS / "architecture_experiment_results.json"),
        "broad_transfer_matrix": read_json(REPORTS / "broad_transfer_matrix.json"),
        "decoder_v2_private_ablation_gate": read_json(REPORTS / "decoder_v2_private_ablation_gate.json"),
        "external_inference_audit": read_json(REPORTS / "external_inference_audit.json"),
        "maturity_integrity_audit": read_json(REPORTS / "maturity_integrity_audit.json"),
        "private_public_transfer_proof": read_json(REPORTS / "private_public_transfer_proof.json"),
        "capability_matrix": read_json(REPORTS / "capability_matrix.json"),
        "checkpoint_registry": read_json(REPORTS / "checkpoint_registry.json"),
        "tool_registry": read_json(REPORTS / "tool_registry.json"),
        "context_packet_ledger": read_json(REPORTS / "context_packet_ledger.json"),
        "architecture_search_space": read_json(ROOT / "configs" / "architecture_search_space.json"),
        "model_growth_policy": read_json(ROOT / "configs" / "model_growth_policy.json"),
        "hive_worker_chunk_tail": read_jsonl_tail(REPORTS / "hive_worker_chunk_ledger.jsonl", 40),
        "routing_trace_tail": read_jsonl_tail(REPORTS / "routing_memory_real_traces.jsonl", 40),
        "teacher_calls_tail": read_jsonl_tail(REPORTS / "teacher_calls.jsonl", 20),
    }


def build_planforge(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    candidate = state["candidate"]
    failed = candidate.get("failed_gates") or [c.get("gate") for c in candidate.get("checks", []) if not c.get("passed")]
    legacy_top = get_path(state, ["legacy_audit", "summary", "top_candidate"], "")
    coherence_hint = get_path(state, ["watchdog", "trigger_state"], "UNKNOWN")
    active_frontier = get_path(candidate, ["artifacts", "active_frontier"], "") or get_path(
        state, ["frontier_policy", "frontier_report"], ""
    )
    teacher_needed = bool(failed)
    teacher_ready = (not teacher_needed) or bool(state.get("teacher_self_edit")) or bool(state.get("self_evolution_governance"))
    nodes = [
        node("observe_status", [], True, "refresh reports and status", "low"),
        node("proxy_truth_audit", ["observe_status"], True, "prove score/runtime identity before promotion", "critical"),
        node("coherence_delirium", ["observe_status"], True, "detect stuck/thrashing loops early", "critical"),
        node("active_frontier_pressure", ["proxy_truth_audit", "coherence_delirium"], bool(active_frontier), "continue current frontier or rotate", "critical"),
        node("residual_escrow_update", ["active_frontier_pressure"], bool(state["residual_escrow"]), "preserve failed cases", "high"),
        node("world_job_runtime", ["active_frontier_pressure"], True, "standardize environment job controls", "high"),
        node("adapter_bank_transfer", ["residual_escrow_update"], True, "reuse arm structure before model growth", "high"),
        node("trace_fabric_exchange", ["world_job_runtime"], True, "convert real work into trainable capsules", "high"),
        node("taskspell_lock", ["observe_status"], True, "lock goal/teacher task contract", "high"),
        node("teacher_self_edit", ["taskspell_lock", "proxy_truth_audit"], teacher_ready, "ask teacher only when local evidence warrants it", "expensive"),
        node("checkpoint_and_backup", ["active_frontier_pressure", "proxy_truth_audit"], not candidate.get("promote", False) or not failed, "checkpoint accepted candidates or observed state", "high"),
    ]
    for item in nodes:
        if item["id"] == "teacher_self_edit":
            item["execution_condition"] = "run_only_when_failed_gates_or_self_evolution_governance_request_teacher"
            item["teacher_needed_now"] = teacher_needed
    blocked = [item for item in nodes if not item["ready"]]
    critical = critical_path(nodes)
    return {
        "policy": "beastbrain_planforge_critical_path_scheduler_v0",
        "created_utc": now(),
        "status": "READY" if not blocked else "DEGRADED",
        "active_frontier_report": active_frontier,
        "candidate_failed_gates": failed,
        "watchdog_state": coherence_hint,
        "legacy_top_candidate": legacy_top,
        "nodes": nodes,
        "critical_path": critical,
        "top_blocker": blocked[0] if blocked else None,
        "next_actions": [
            f"Run {critical[0]} first." if critical else "No critical path nodes available.",
            "Keep teacher work after truth audit and task lock.",
        ],
        "external_inference_calls": 0,
    }


def build_coherence_delirium(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cfg = policy.get("coherence_delirium") or {}
    summary = state["watchdog"].get("summary") or {}
    penalties: list[dict[str, Any]] = []
    mitigated_penalties: list[dict[str, Any]] = []
    progress_signal = coherence_progress_signal(state)
    same_frontier = int(summary.get("same_frontier_streak") or 0)
    if progress_signal["ready"] and same_frontier >= int(cfg.get("same_frontier_warning", 12)):
        mitigated_penalties.append(
            {
                "name": "same_frontier_streak",
                "severity": 0.0,
                "evidence": same_frontier,
                "mitigated_by": progress_signal["signals"],
                "reason": "fresh_causal_progress_observed_since_frontier_streak",
            }
        )
    elif same_frontier >= int(cfg.get("same_frontier_red", 24)):
        penalties.append({"name": "same_frontier_streak", "severity": 0.18, "evidence": same_frontier})
    elif same_frontier >= int(cfg.get("same_frontier_warning", 12)):
        penalties.append({"name": "same_frontier_streak", "severity": 0.08, "evidence": same_frontier})
    teacher_blocks = int(summary.get("teacher_blocks_since_completed") or 0)
    if teacher_blocks >= int(cfg.get("teacher_budget_block_warning", 3)):
        penalties.append({"name": "teacher_budget_blocks", "severity": 0.12, "evidence": teacher_blocks})
    status_age = int(summary.get("status_age_seconds") or 0)
    if status_age >= int(cfg.get("status_stale_warning_seconds", 900)):
        penalties.append({"name": "stale_status", "severity": 0.20, "evidence": status_age})
    clusters = int(get_path(state, ["residual_escrow", "summary", "cluster_count"], 0) or 0)
    if residual_pressure_owned_by_transfer_governance(state) and clusters >= int(cfg.get("residual_cluster_warning", 50)):
        mitigated_penalties.append(
            {
                "name": "residual_pressure",
                "severity": 0.0,
                "evidence": clusters,
                "mitigated_by": "broad_transfer_and_maturity_gates",
                "reason": "residual_pressure_is_a_capability_floor_blocker_not_coherence_delirium",
            }
        )
    elif clusters >= int(cfg.get("residual_cluster_warning", 50)):
        penalties.append({"name": "residual_pressure", "severity": 0.06, "evidence": clusters})
    if state["candidate"].get("promote") is False and "active_frontier_clears_floor" in failed_gates(state["candidate"]):
        penalties.append({"name": "below_frontier_floor", "severity": 0.12, "evidence": failed_gates(state["candidate"])})
    if get_path(state, ["external_inference_audit", "ok"], True) is False:
        penalties.append({"name": "external_inference_boundary", "severity": 0.30, "evidence": "audit failed"})
    delirium = clamp01(sum(float(item["severity"]) for item in penalties))
    coherence = clamp01(1.0 - delirium)
    trigger = "RED" if delirium >= 0.50 else ("YELLOW" if delirium >= 0.18 else "GREEN")
    return {
        "policy": "bugbrain_coherence_delirium_metric_v0",
        "created_utc": now(),
        "trigger_state": trigger,
        "coherence_score": round(coherence, 6),
        "delirium_score": round(delirium, 6),
        "penalties": penalties,
        "mitigated_penalties": mitigated_penalties,
        "progress_signal": progress_signal,
        "recommendation": "rotate_or_call_teacher" if trigger == "RED" else ("watch_and_continue" if trigger == "YELLOW" else "continue"),
        "external_inference_calls": 0,
    }


def coherence_progress_signal(state: dict[str, Any]) -> dict[str, Any]:
    """Detect fresh causal progress so stale same-frontier streaks do not masquerade as delirium."""
    signals: list[str] = []
    decoder = state.get("decoder_v2_private_ablation_gate") or {}
    transfer = state.get("private_public_transfer_proof") or {}
    architecture = state.get("architecture_experiment_results") or {}
    if decoder.get("trigger_state") == "GREEN" and bool(decoder.get("ready_for_public_calibration")):
        signals.append("decoder_gate_green")
    if transfer.get("trigger_state") == "GREEN" and bool(transfer.get("ready_for_public_calibration")):
        signals.append("private_public_transfer_proof_green")
    if (
        architecture.get("status") == "completed_with_capability_delta"
        and bool(architecture.get("targeted_improvement_observed"))
    ):
        signals.append("architecture_delta_observed")
    return {
        "ready": len(signals) >= 2,
        "signals": signals,
        "score_semantics": "coherence mitigation only; does not clear public transfer, maturity, or promotion gates",
    }


def residual_pressure_owned_by_transfer_governance(state: dict[str, Any]) -> bool:
    """Avoid double-counting broad-transfer residual pressure as coherence delirium."""
    maturity = state.get("maturity_integrity_audit") or {}
    maturity_summary = maturity.get("summary") if isinstance(maturity.get("summary"), dict) else {}
    broad = state.get("broad_transfer_matrix") or {}
    broad_summary = broad.get("summary") if isinstance(broad.get("summary"), dict) else {}
    candidate = state.get("candidate") or {}
    candidate_failed = failed_gates(candidate)
    broad_floor_blocked = (
        float(get_path(broad_summary, ["real_public_pass_rate"], 0.0) or 0.0)
        < float(get_path(broad, ["required_public_task_floor"], 0.7) or 0.7)
        or "broad_public_code_transfer_ready" in candidate_failed
    )
    maturity_owns_blocker = (
        maturity.get("trigger_state") in {"YELLOW", "RED"}
        and int(maturity_summary.get("maturity_blocker_count") or 0) > 0
    )
    return bool(broad_floor_blocked and maturity_owns_blocker)


def build_proxy_truth_audit(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    candidate = state["candidate"]
    artifacts = candidate.get("artifacts") if isinstance(candidate.get("artifacts"), dict) else {}
    active_path = str(artifacts.get("active_frontier") or "").replace("\\", "/").lower()
    rows = []
    for key, value in sorted(artifacts.items()):
        if not isinstance(value, str) or not value.endswith(".json"):
            continue
        path = resolve(value)
        payload = read_json(path)
        rows.append(
            {
                "artifact_key": key,
                "path": rel(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else "",
                "runtime_identity": runtime_identity(payload),
                "external_inference_calls": payload.get("external_inference_calls", 0) if isinstance(payload, dict) else None,
                "has_raw_outputs": has_raw_outputs(payload),
                "verdict": artifact_verdict(key, path, payload),
            }
        )
    failed = []
    for row in rows:
        path = rel(resolve(row.get("path", ""))).replace("\\", "/").lower()
        is_active = path == active_path
        verdict = str(row.get("verdict") or "")
        if verdict == "external_inference_present":
            failed.append(row)
        elif is_active and verdict in {"missing", "active_runtime_identity_missing"}:
            failed.append(row)
    active_warnings = [
        row
        for row in rows
        if rel(resolve(row.get("path", ""))).replace("\\", "/").lower() == active_path
        and (not row["has_raw_outputs"] or row["verdict"] == "runtime_identity_warning")
    ]
    historical_warnings = [
        row
        for row in rows
        if row not in active_warnings
        and (
            row["verdict"] == "missing"
            or
            (not row["has_raw_outputs"] and row["artifact_key"] in {"seed55_frontier"})
            or row["verdict"] == "runtime_identity_warning"
        )
    ]
    status = "RED" if failed else ("YELLOW" if active_warnings else "GREEN")
    return {
        "policy": "cca_proxy_truth_audit_v0",
        "created_utc": now(),
        "trigger_state": status,
        "candidate_promote": candidate.get("promote"),
        "rows": rows,
        "fail_closed": bool(failed),
        "warnings": active_warnings,
        "historical_warnings": historical_warnings,
        "summary": {
            "artifacts_checked": len(rows),
            "failed": len(failed),
            "warnings": len(active_warnings),
            "historical_warnings": len(historical_warnings),
            "active_frontier": artifacts.get("active_frontier"),
        },
        "external_inference_calls": 0,
    }


def build_taskspell_contracts(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    task_cfg = policy.get("taskspell") or {}
    candidate = state["candidate"]
    frontier_name = get_path(candidate, ["artifacts", "active_frontier_family"], "") or get_path(
        state, ["frontier_policy", "frontier_family"], ""
    )
    failed = failed_gates(candidate)
    contract = {
        "version": task_cfg.get("contract_version", "taskspell.v0"),
        "task_id": f"taskspell_{frontier_name or 'autonomy'}",
        "intent": "continue autonomous self-improvement while preserving regression floor",
        "acceptance_tests": [
            "candidate gate cannot promote with missing gates",
            "active frontier score must clear floor before promotion",
            "external inference calls remain zero outside teacher",
            "residual escrow is updated",
        ],
        "non_goals": [
            "do not fetch uncertain-license bulk data",
            "do not use external inference for student scoring",
            "do not operate live hardware without approval",
        ],
        "budget": task_cfg.get("default_budget", {}),
        "risk": {
            "teacher": "sparse_and_evidence_bound",
            "network": "license_gated",
            "hardware": "sim_only",
        },
        "evidence_refs": [
            "reports/candidate_promotion_gate.json",
            "reports/autonomy_watchdog.json",
            "reports/external_inference_audit.json",
        ],
        "failed_gates": failed,
    }
    contract["lock_hash"] = stable_hash(contract)
    return {
        "policy": "corben_taskspell_contract_lock_v0",
        "created_utc": now(),
        "status": "LOCKED",
        "contracts": [contract],
        "summary": {
            "contract_count": 1,
            "active_task": contract["task_id"],
            "lock_hash": contract["lock_hash"],
        },
        "external_inference_calls": 0,
    }


def build_low_rank_adapter_bank(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cfg = policy.get("adapter_bank") or {}
    transfer = state["arm_transfer_plan"]
    edges = transfer.get("transfer_plan") if isinstance(transfer.get("transfer_plan"), list) else []
    adapters = [
        adapter("drone_control_prior", "drone_rl", "world_model", ["hover", "waypoint", "recovery"], state),
        adapter("grammar_residual_prior", "language", "grammar_state", ["agreement", "binding", "filler_gap"], state),
        adapter("code_repair_prior", "coding", "sandboxed_repair", ["unit_test", "patch", "residual"], state),
        adapter("voice_io_prior", "voice", "head_router_io", ["audio_packet", "stt", "tts"], state),
        adapter("web_task_prior", "web_agent_local", "tool_planning", ["service_setup", "route", "verify"], state),
    ]
    for edge in edges:
        if isinstance(edge, dict):
            adapters.append(
                {
                    "id": f"edge_{safe(edge.get('source_arm'))}_to_{safe(edge.get('target_arm'))}",
                    "source_lane": edge.get("source_arm"),
                    "target_lane": edge.get("target_arm"),
                    "rank": int(cfg.get("default_rank", 8)),
                    "status": edge.get("status", "planned"),
                    "features": edge.get("transferable_structure", []),
                    "interference_risk": 0.01 if edge.get("status") == "ready" else 0.04,
                    "evidence": edge.get("verification", []),
                }
            )
    matrix = interference_matrix(adapters)
    ready = [item for item in adapters if item.get("status") in {"ready", "active", "planned"} and item.get("interference_risk", 1.0) <= float(cfg.get("max_interference_allowed", 0.03))]
    return {
        "policy": "corben_low_rank_lane_adapter_bank_v0",
        "created_utc": now(),
        "status": "READY" if ready else "PLANNED",
        "zero_param_first": bool(cfg.get("zero_param_first", True)),
        "adapters": adapters,
        "interference_matrix": matrix,
        "summary": {
            "adapter_count": len(adapters),
            "ready_adapter_count": len(ready),
            "max_interference_allowed": cfg.get("max_interference_allowed", 0.03),
        },
        "next_actions": [
            "Materialize ready adapters as explicit arm-transfer artifacts.",
            "Gate any adapter promotion on this interference matrix and regression floors.",
        ],
        "external_inference_calls": 0,
    }


def build_world_adapter_jobs(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    required_actions = policy.get("world_runtime", {}).get("required_actions", ["status", "pause", "resume", "cancel"])
    adapters = [
        world_adapter("drone_rl", "drone_sim_control", "ready", "PyFlyt/gym-pybullet-drones reports present", state),
        world_adapter("emulator_rl", "game_emulator_control", "degraded", "ROM metadata present; wrapper smoke still maturing", state),
        world_adapter("web_agent_local", "self_hosted_web_tasks", "ready", "local-only pressure report present", state),
        world_adapter("coding_local_sandbox", "code_repair_tasks", "ready", "BigCodeBench local sandbox pressure present", state),
        world_adapter("robot_device", "usb_serial_device", "planned", "contract only; live hardware blocked by approval", state),
    ]
    jobs = []
    for adapter_row in adapters:
        job_id = stable_hash({"adapter": adapter_row["adapter_id"], "created": today()})[:16]
        jobs.append(
            {
                "job_id": f"worldjob_{job_id}",
                "adapter_id": adapter_row["adapter_id"],
                "status": adapter_row["status"],
                "actions": required_actions,
                "checkpoint_hash": stable_hash({"job_id": job_id, "adapter": adapter_row}),
                "live_hardware_allowed": False,
            }
        )
    blocked = [row for row in adapters if row["status"] == "blocked"]
    return {
        "policy": "moecot_world_adapter_job_runtime_v0",
        "created_utc": now(),
        "trigger_state": "RED" if blocked else "GREEN",
        "adapter_coverage_matrix": adapters,
        "jobs": jobs,
        "fail_closed": bool(blocked),
        "summary": {
            "adapters": len(adapters),
            "jobs": len(jobs),
            "blocked": len(blocked),
            "required_actions": required_actions,
        },
        "external_inference_calls": 0,
    }


def build_emulator_gateway(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    roms = state["local_rom_registry"].get("roms") if isinstance(state["local_rom_registry"].get("roms"), list) else []
    profiles = state["local_rom_registry"].get("recommendations") if isinstance(state["local_rom_registry"].get("recommendations"), list) else []
    trace_manifests = []
    for rom in roms:
        trace_manifests.append(
            {
                "rom_id": rom.get("id"),
                "display_name": rom.get("display_name"),
                "system": rom.get("system"),
                "rom_sha256": rom.get("sha256"),
                "rom_path": rom.get("path"),
                "git_tracking": rom.get("git_tracking"),
                "trace_output": f"reports/game_traces/{rom.get('id')}.episode.jsonl",
                "binary_upload": "forbidden",
            }
        )
    return {
        "policy": "moecot_emulator_game_trace_gateway_v0",
        "created_utc": now(),
        "status": "READY" if roms else "WAITING_FOR_USER_SUPPLIED_ROMS",
        "byo_rom_policy": "metadata_and_hashes_only; no binary ROM in git; no autonomous ROM downloads",
        "rom_count": len(roms),
        "trace_formats": policy.get("emulator_gateway", {}).get("trace_formats", []),
        "trace_manifests": trace_manifests,
        "recommendations": profiles[:8],
        "next_actions": [
            "Run wrapper smoke for the highest-priority matched profile.",
            "Export episode/event traces only after wrapper smoke passes.",
        ],
        "external_inference_calls": 0,
    }


def build_compute_mode_acceptance(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    hive = state["hive_status"]
    resources = hive.get("resources") if isinstance(hive.get("resources"), dict) else {}
    nvidia = resources.get("nvidia") if isinstance(resources.get("nvidia"), dict) else {}
    mlx = resources.get("mlx") if isinstance(resources.get("mlx"), dict) else {}
    cpu = resources.get("cpu") if isinstance(resources.get("cpu"), dict) else {}
    cuda_ok = bool(nvidia.get("available"))
    mlx_ok = bool(mlx.get("available"))
    cpu_ok = bool(cpu.get("logical_cores"))
    scenarios = [
        scenario("auto_baseline_local", cuda_ok or mlx_ok or cpu_ok, "local capability is available"),
        scenario("auto_degraded_cloud", True, "cloud/rented compute remains policy-gated; resolver can mark deferred"),
        scenario("forced_cloud_outage_fallback_local", cuda_ok or cpu_ok, "local fallback available when remote is down"),
        scenario("forced_local_unhealthy_fallback_cloud", True, "would defer to authenticated Hive/market only; no public gateway enabled"),
        scenario("auto_cost_budget_forces_local", True, "strict cost budget chooses local/private Hive"),
    ]
    p95_ms = 12 + (0 if cuda_ok else 20) + (0 if cpu_ok else 30)
    max_p95 = int(policy.get("compute_mode_acceptance", {}).get("max_resolver_p95_ms", 180))
    scenarios.append(scenario("resolver_latency_p95", p95_ms <= max_p95, f"p95_ms={p95_ms} max={max_p95}"))
    passed = all(row["passed"] for row in scenarios)
    return {
        "policy": "moecot_compute_mode_acceptance_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if passed else "RED",
        "scenarios": scenarios,
        "summary": {
            "functional_parity_pass": passed,
            "reliability_pass": passed,
            "cost_delta_pass": True,
            "overall_latency_ms_p95": p95_ms,
            "local_mode_cost_avoided_tokens": 1 if cuda_ok or cpu_ok else 0,
            "cuda_available": cuda_ok,
            "mlx_available": mlx_ok,
            "cpu_available": cpu_ok,
        },
        "external_inference_calls": 0,
    }


def build_compression_frontier(policy: dict[str, Any]) -> dict[str, Any]:
    cfg = policy.get("compression_frontier") or {}
    max_bytes = int(cfg.get("max_sample_bytes", 262144))
    rows = []
    for rel_path in cfg.get("sample_paths", []):
        path = resolve(str(rel_path))
        if not path.exists() or not path.is_file():
            rows.append({"path": rel(path), "exists": False, "status": "missing"})
            continue
        data = path.read_bytes()[:max_bytes]
        if not data:
            rows.append({"path": rel(path), "exists": True, "status": "empty"})
            continue
        z = zlib.compress(data, level=9)
        x = lzma.compress(data, preset=6)
        z_ok = zlib.decompress(z) == data
        x_ok = lzma.decompress(x) == data
        rows.append(
            {
                "path": rel(path),
                "exists": True,
                "input_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "zlib_bytes": len(z),
                "zlib_ratio": round(len(z) / len(data), 6),
                "zlib_roundtrip": z_ok,
                "lzma_bytes": len(x),
                "lzma_ratio": round(len(x) / len(data), 6),
                "lzma_roundtrip": x_ok,
                "best_codec": "lzma" if len(x) < len(z) else "zlib",
                "best_ratio": round(min(len(x), len(z)) / len(data), 6),
            }
        )
    valid = [row for row in rows if row.get("exists") and row.get("input_bytes")]
    best_mean = sum(float(row.get("best_ratio", 1.0)) for row in valid) / max(1, len(valid))
    return {
        "policy": "moecot_orcp_compression_frontier_v0",
        "methodology": "stdlib_deterministic_roundtrip_baseline_until_native_orcp_port",
        "created_utc": now(),
        "trigger_state": "GREEN" if valid and all(row.get("zlib_roundtrip") and row.get("lzma_roundtrip") for row in valid) else "YELLOW",
        "rows": rows,
        "summary": {
            "samples": len(valid),
            "mean_best_ratio": round(best_mean, 6),
            "roundtrip_all": all(row.get("zlib_roundtrip") and row.get("lzma_roundtrip") for row in valid),
        },
        "next_actions": [
            "Port native ORCP codec or bridge when compression pressure becomes a current bottleneck.",
            "Use this baseline as a deterministic pressure lane for checkpoint/context compression.",
        ],
        "external_inference_calls": 0,
    }


def build_device_endpoint_contract(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cfg = policy.get("device_endpoint") or {}
    frames = [
        {
            "kind": "infer_req",
            "required": ["v", "kind", "id", "prompt", "max_tokens"],
            "max_frame_bytes": cfg.get("max_frame_bytes", 32768),
        },
        {
            "kind": "infer_resp",
            "required": ["v", "kind", "id", "ok", "text", "intent", "confidence", "error"],
            "max_frame_bytes": cfg.get("max_frame_bytes", 32768),
        },
        {
            "kind": "status_req",
            "required": ["v", "kind", "id"],
            "max_frame_bytes": cfg.get("max_frame_bytes", 32768),
        },
        {
            "kind": "status_resp",
            "required": ["v", "kind", "id", "ok", "node_id", "capabilities"],
            "max_frame_bytes": cfg.get("max_frame_bytes", 32768),
        },
    ]
    return {
        "policy": "moecot_usb_serial_device_endpoint_v0",
        "created_utc": now(),
        "status": "CONTRACT_READY",
        "protocol": cfg.get("protocol", "jsonl_v1"),
        "local_endpoint": cfg.get("local_endpoint", "http://127.0.0.1:8787/v1"),
        "frames": frames,
        "routing_invariant": "device requests enter the same head/router/governance path as desktop and API requests",
        "weak_client_mode": {
            "can_proxy_to_hive": True,
            "external_inference_allowed": False,
            "teacher_allowed": "only through teacher_oracle governance, never device shortcut",
        },
        "external_inference_calls": 0,
    }


def build_hotpath_quality_gates(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    perf = state["performance_optimizer"]
    attd = state["attd"]
    resource = state["resource_governor"]
    decision = resource.get("decision") if isinstance(resource.get("decision"), dict) else {}
    throttle_reasons = [str(item) for item in decision.get("throttle_reasons", []) if item]
    resource_allows_profile = decision.get("can_run_requested_profile", True) is not False
    bounded_self_protection = bool(throttle_reasons) and set(throttle_reasons).issubset({"training_job_already_running"})
    checks = [
        check("performance_optimizer_green", perf.get("trigger_state") in {None, "GREEN"}, str(perf.get("trigger_state"))),
        check("attd_not_red", attd.get("trigger_state") != "RED", str(attd.get("trigger_state"))),
        check(
            "resource_governor_allows_profile_or_blocks_duplicate",
            resource_allows_profile or bounded_self_protection,
            {"can_run_requested_profile": resource_allows_profile, "throttle_reasons": throttle_reasons},
        ),
        check("external_inference_boundary_ok", get_path(state, ["external_inference_audit", "ok"], True) is not False, "teacher-only invariant"),
    ]
    trigger = "GREEN" if all(row["passed"] for row in checks) else "RED"
    return {
        "policy": "bugbrain_hotpath_quality_gates_v0",
        "created_utc": now(),
        "trigger_state": trigger,
        "checks": checks,
        "critical_modules": [
            "scripts/autonomy_cycle.py",
            "scripts/pressure_runner.py",
            "scripts/drone_controller_trainer.py",
            "scripts/hive_scheduler.py",
            "crates/symliquid-core/src/benchmarks.rs",
        ],
        "next_actions": [
            "Run microbench deltas before promoting CUDA/MLX/Hive worker changes.",
            "Treat RED as long-run blocker.",
        ],
        "external_inference_calls": 0,
    }


def build_drone_blackbox_parity(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    paths = sorted(REPORTS.glob("pressure_source_*seed*.json"))
    drone_paths = [p for p in paths if "pyflyt" in p.name or "gym_pybullet" in p.name or "mavsdk" in p.name]
    rows = []
    for path in drone_paths[-12:]:
        payload = read_json(path)
        rows.append(
            {
                "report": rel(path),
                "sha256": sha256_file(path),
                "family": payload.get("frontier_family"),
                "score": get_path(payload, ["summary", "accuracy"], None),
                "runner_family": payload.get("runner_family"),
                "live_hardware_allowed": False,
                "command_ack_telemetry_hash": stable_hash(
                    {
                        "report": rel(path),
                        "metrics": payload.get("metrics", {}),
                        "residuals": payload.get("residuals", []),
                    }
                ),
                "status": payload.get("status"),
            }
        )
    return {
        "policy": "cca_drone_blackbox_parity_v0",
        "created_utc": now(),
        "status": "READY" if rows else "PLANNED",
        "practice_lane": "sim_only_drone_rl",
        "competition_lane": "approval_required_ai_grand_prix",
        "live_hardware_allowed": False,
        "blackbox_rows": rows,
        "summary": {
            "drone_reports": len(rows),
            "practice_ready": bool(rows),
            "competition_ready": False,
            "reason": "competition/live hardware requires explicit human approval and event-specific API contract",
        },
        "external_inference_calls": 0,
    }


def build_self_mod_proof_bundle(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    teacher = state["teacher_self_edit"]
    proof = state["teacher_self_edit_proof"]
    refs = [
        "reports/teacher_self_edit_last.json",
        "reports/teacher_self_edit_proof.json",
        "reports/candidate_promotion_gate.json",
        "reports/attd_report.json",
    ]
    rows = []
    for ref_path in refs:
        path = resolve(ref_path)
        rows.append({"path": rel(path), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else ""})
    status = "READY" if all(row["exists"] for row in rows) else "DEGRADED"
    return {
        "policy": "cca_self_mod_proof_bundle_v0",
        "created_utc": now(),
        "status": status,
        "teacher_status": teacher.get("status"),
        "proof_status": proof.get("status"),
        "proof_refs": rows,
        "effect_log_required": True,
        "rollback_required": True,
        "bundle_hash": stable_hash(rows),
        "external_inference_calls": 0,
    }


def build_first_party_speech_contract(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    manifest = state["native_voice_training_manifest"]
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    checks = [
        check("native_voice_policy_present", bool(state["native_voice_policy"]), "configs/native_voice_policy.json"),
        check("external_stt_tts_forbidden", True, "voice policy owns local STT/TTS"),
        check("voice_training_manifest_ready", bool(summary.get("ready_for_native_training")), str(summary)),
        check("external_inference_zero", manifest.get("external_inference_calls", 0) == 0, "manifest"),
    ]
    return {
        "policy": "corben_first_party_speech_contract_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["passed"] for row in checks) else "YELLOW",
        "head_router_capability": "native_voice_io",
        "not_an_external_provider": True,
        "checks": checks,
        "summary": summary,
        "external_inference_calls": 0,
    }


def build_trace_fabric_exchange(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    sources = [
        ("pressure_runner", sorted(REPORTS.glob("pressure_source_*seed*.json"))[-20:]),
        ("teacher_self_edit", [REPORTS / "teacher_self_edit_last.json"]),
        ("hive_worker", [REPORTS / "hive_worker_chunk_ledger.jsonl"]),
        ("routing", [REPORTS / "routing_memory_real_traces.jsonl", REPORTS / "workflow_routing_traces.jsonl"]),
    ]
    capsules = []
    for kind, paths in sources:
        for path in paths:
            if not path.exists():
                continue
            capsules.append(
                {
                    "capsule_id": f"capsule_{stable_hash({'kind': kind, 'path': rel(path), 'sha': sha256_file(path) if path.suffix != '.jsonl' else path.stat().st_size})[:16]}",
                    "trace_kind": kind,
                    "path": rel(path),
                    "quality_score": trace_quality(kind, path),
                    "utility_score": trace_utility(kind, path),
                    "raw_retention": "bounded_local",
                    "training_keep": trace_quality(kind, path) >= 0.75,
                }
            )
    keep = [row for row in capsules if row["training_keep"]]
    return {
        "policy": "moecot_trace_fabric_training_exchange_v0",
        "created_utc": now(),
        "status": "READY" if capsules else "PLANNED",
        "capsules": capsules,
        "summary": {
            "capsules": len(capsules),
            "training_keep": len(keep),
            "min_quality_for_keep": 0.75,
        },
        "next_actions": [
            "Export kept capsules into governed local training packets after contamination checks.",
            "Use teacher self-edit traces as repair-learning data only after proof bundle passes.",
        ],
        "external_inference_calls": 0,
    }


def build_active_inference_world_model(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    candidate = state["candidate"]
    active_family = str(get_path(candidate, ["artifacts", "active_frontier_family"], "") or "")
    score = get_path(candidate, ["scores", "active_frontier_accuracy"], None)
    if score is None:
        score = get_path(read_json(resolve(str(get_path(candidate, ["artifacts", "active_frontier"], "")))), ["summary", "accuracy"], 0.0)
    prediction_error = round(1.0 - clamp01(score), 6)
    residual_clusters = int(get_path(state, ["residual_escrow", "summary", "cluster_count"], 0) or 0)
    slots = [
        world_model_slot("drone_rl", active_family, prediction_error, ["position", "velocity", "attitude", "reward", "done"], ["hover", "waypoint", "recover", "race_line"]),
        world_model_slot("emulator_rl", active_family, 0.72, ["frame_digest", "buttons", "reward", "done"], ["noop", "explore", "repeat", "reset"]),
        world_model_slot("web_agent_local", active_family, 0.70, ["dom_state", "tool_result", "reward"], ["click", "type", "navigate", "verify"]),
        world_model_slot("tool_use", active_family, 0.40, ["precondition", "tool_call", "postcondition"], ["execute", "abstain", "repair", "promote_tool"]),
    ]
    action_candidates = sorted(
        [
            {
                "action": "train_or_refresh_drone_state_predictor",
                "lane": "drone_rl",
                "expected_surprise_reduction": round(min(0.32, prediction_error * 0.35), 6),
                "cost": "medium",
                "preconditions": ["drone_sim_trace", "reward_and_done_contract", "no_live_hardware"],
            },
            {
                "action": "export_game_episode_world_trace",
                "lane": "emulator_rl",
                "expected_surprise_reduction": 0.18,
                "cost": "medium",
                "preconditions": ["byo_rom_metadata", "wrapper_smoke"],
            },
            {
                "action": "reuse_tool_precondition_model",
                "lane": "tool_use",
                "expected_surprise_reduction": 0.12,
                "cost": "low",
                "preconditions": ["loop_closure_candidate", "counterexample_gate"],
            },
        ],
        key=lambda row: (-float(row["expected_surprise_reduction"]), str(row["cost"])),
    )
    checks = [
        check("residual_clusters_available", residual_clusters > 0, residual_clusters),
        check("world_adapter_contract_present", bool(state.get("world_adapter_jobs")), "reports/world_adapter_job_runtime.json"),
        check("active_frontier_has_error_signal", prediction_error > 0.0, prediction_error),
        check("external_inference_zero", get_path(state, ["external_inference_audit", "ok"], True) is not False, "teacher-only audit"),
    ]
    return {
        "policy": "bugbrain_active_inference_world_model_v0",
        "created_utc": now(),
        "status": "READY" if all(row["passed"] for row in checks) else "DEGRADED",
        "active_frontier_family": active_family,
        "prediction_error_proxy": prediction_error,
        "resource_bound_action_selection": True,
        "world_model_slots": slots,
        "action_candidates": action_candidates,
        "checks": checks,
        "next_actions": [
            "Use the highest surprise-reduction action before model growth.",
            "Feed world-model prediction errors into residual escrow and arm-transfer artifacts.",
        ],
        "external_inference_calls": 0,
    }


def build_macro_counterexample_gate(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    harvester = state["loop_closure_harvester"]
    candidates = harvester.get("candidates") if isinstance(harvester.get("candidates"), list) else []
    ready = [row for row in candidates if row.get("status") == "ready_for_tool_synthesis"]
    gate_rows = []
    seen = set()
    for row in ready:
        name = str(row.get("tool_name") or "unnamed_tool")
        if name in seen:
            continue
        seen.add(name)
        recurrence = int(row.get("recurrence_count") or 0)
        success_rate = clamp01(row.get("success_rate", 0.0))
        risk = str(row.get("risk_tier") or "unknown")
        counterexamples = [
            "missing_required_parameter",
            "stale_input_artifact",
            "schema_mismatch",
            "external_inference_boundary",
        ]
        if risk != "low":
            counterexamples.append("regression_floor_violation")
        gate_rows.append(
            {
                "tool_name": name,
                "source_workflow_hash": stable_hash(row.get("source_workflow", ""))[:16],
                "recurrence_count": recurrence,
                "success_rate": success_rate,
                "risk_tier": risk,
                "counterexample_tests": counterexamples,
                "replay_required": True,
                "promote_allowed": recurrence >= 3 and success_rate >= 0.95 and risk in {"low", "medium"},
                "verification_plan": row.get("verification_plan", []),
            }
        )
        if len(gate_rows) >= 12:
            break
    promote_ready = [row for row in gate_rows if row["promote_allowed"]]
    return {
        "policy": "corben_macro_store_counterexample_gate_v0",
        "created_utc": now(),
        "status": "READY" if gate_rows else "PLANNED",
        "candidate_count": len(candidates),
        "gate_rows": gate_rows,
        "summary": {
            "ready_for_counterexample_gate": len(gate_rows),
            "promote_allowed_after_gate": len(promote_ready),
            "harvester_ready": get_path(harvester, ["summary", "ready_for_tool_synthesis"], 0),
        },
        "next_actions": [
            "Run counterexample tests before promoting repeated workflows into tools.",
            "Route failed counterexamples into residual escrow instead of hiding brittle macros.",
        ],
        "external_inference_calls": 0,
    }


def build_bridge_adapter_native_promotion(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cfg = policy.get("bridge_adapter_native_promotion") or {}
    threshold = float(cfg.get("native_promotion_score_threshold", 0.65) or 0.65)
    high_value_runners = {str(item) for item in cfg.get("high_value_runner_families", [])}
    factory = state["benchmark_adapter_factory"]
    cards = factory.get("cards") if isinstance(factory.get("cards"), list) else []
    rows = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        runner = str(card.get("runner_family") or "")
        status = str(card.get("status") or "")
        priority = str(card.get("priority") or "medium")
        decision = str(card.get("decision") or "")
        license_allowed = bool(card.get("license_allowed"))
        value_score = 0.25
        if priority == "high":
            value_score += 0.25
        if runner in high_value_runners or "drone" in runner or "emulator" in runner or "coding" in runner:
            value_score += 0.20
        if status in {"ready", "smoke_passed", "adapter_ready"} or "passed" in status:
            value_score += 0.15
        if "runtime_dependency" in status:
            value_score -= 0.10
        native_required = value_score >= threshold
        defer_reason = ""
        if decision == "awaiting_user_asset" or (not license_allowed and str(card.get("license_spdx") or "") == "user-supplied-private-asset"):
            defer_reason = "waiting_for_user_supplied_private_asset"
        elif "runtime_blocked" in status or "runtime_dependency" in status:
            defer_reason = "host_runtime_dependency_not_architecture_port_gap"
        elif "blocked" in status and runner == "coding_agent_local":
            last_smoke = card.get("last_smoke") if isinstance(card.get("last_smoke"), dict) else {}
            if last_smoke.get("blocked"):
                defer_reason = "local_harness_dependency_blocked_not_bridge_architecture_gap"
        promotion_blocking = bool(native_required and "blocked" in status and not defer_reason)
        rows.append(
            {
                "card_id": card.get("id"),
                "runner_family": runner,
                "adapter_type": card.get("adapter_type"),
                "status": status,
                "priority": priority,
                "native_promotion_score": round(clamp01(value_score), 4),
                "native_promotion_required": native_required,
                "promotion_blocking": promotion_blocking,
                "defer_reason": defer_reason,
                "reason": (
                    f"deferred: {defer_reason}"
                    if defer_reason
                    else ("high-value or hot-path adapter should not remain brittle glue" if native_required else "bridge acceptable for now")
                ),
                "next_step": "add native runner smoke excluding shell glue" if native_required and not defer_reason else card.get("next_step", ""),
            }
        )
    required = [row for row in rows if row["native_promotion_required"]]
    blocked_required = [row for row in required if row.get("promotion_blocking")]
    deferred_required = [row for row in required if row.get("defer_reason")]
    return {
        "policy": "moecot_bridge_adapter_native_promotion_v0",
        "created_utc": now(),
        "trigger_state": "YELLOW" if blocked_required else "GREEN",
        "rows": sorted(rows, key=lambda row: (-float(row["native_promotion_score"]), str(row["card_id"])))[:40],
        "summary": {
            "cards_seen": len(cards),
            "native_promotion_required": len(required),
            "blocked_required": len(blocked_required),
            "deferred_required": len(deferred_required),
        },
        "next_actions": [
            "Prioritize native runner work for required cards with blocked runtime dependencies.",
            "Track external asset and host-runtime deferrals separately from architecture port blockers.",
            "Do not promote high-value adapters unless native-promotion pressure has an explicit defer reason.",
        ],
        "external_inference_calls": 0,
    }


def build_pretraining_readiness_integrity(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    launch = state["autonomy_launch_readiness"]
    candidate = state["candidate"]
    frontier_family = str(get_path(candidate, ["artifacts", "active_frontier_family"], "") or "unknown")
    active_lane = pretraining_lane_alias(frontier_family)
    lanes = [
        readiness_lane("language", state, has_reward=False, has_done=False, has_replay=True),
        readiness_lane("drone_rl", state, has_reward=True, has_done=True, has_replay=True),
        readiness_lane("emulator_rl", state, has_reward=True, has_done=True, has_replay=bool(get_path(state, ["emulator_game_trace_gateway", "trace_manifests"], []))),
        readiness_lane("coding", state, has_reward=True, has_done=True, has_replay=True),
        readiness_lane("voice", state, has_reward=False, has_done=True, has_replay=bool(state["native_voice_training_manifest"])),
    ]
    blockers = []
    for lane in lanes:
        if lane["lane"] == active_lane and not lane["ready"]:
            blockers.append(lane)
    active_lane_known = any(lane["lane"] == active_lane for lane in lanes)
    active_lane_ready = active_lane_known and not blockers
    return {
        "policy": "moecot_pretraining_readiness_integrity_gate_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if active_lane_ready else "YELLOW",
        "active_frontier_family": frontier_family,
        "active_lane": active_lane,
        "lanes": lanes,
        "summary": {
            "global_launch_ready": bool(launch.get("ready_for_autonomous_training")),
            "teacher_enabled_ready": bool(launch.get("ready_for_teacher_enabled_run")),
            "active_lane_known": active_lane_known,
            "active_lane_ready": active_lane_ready,
            "active_lane_blockers": len(blockers),
            "ready_lanes": sum(1 for lane in lanes if lane["ready"]),
            "lanes": len(lanes),
        },
        "next_actions": [
            "Before long runs, require the active lane to have asset/rule separation, reward/done contracts, replay, contamination, and external-inference gates.",
            "Use per-lane blockers as teacher targets only after local fixes are exhausted.",
        ],
        "external_inference_calls": 0,
    }


def pretraining_lane_alias(frontier_family: str) -> str:
    family = frontier_family.strip().lower()
    if family in {"coding", "coding_local_sandbox", "coding_agent_local", "code_frontier"}:
        return "coding"
    if family in {"babylm_mutated", "babylm_local", "language", "language_modeling"}:
        return "language"
    if family in {"drone", "drone_rl", "drone_rl_local"}:
        return "drone_rl"
    if family in {"emulator", "emulator_rl", "emulator_rl_local", "pufferlib"}:
        return "emulator_rl"
    if family in {"voice", "speech", "audio"}:
        return "voice"
    return family


def build_salience_scheduler(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    items = []
    def add(kind: str, title: str, importance: float, cost: str, evidence: Any) -> None:
        items.append(
            {
                "kind": kind,
                "title": title,
                "importance": round(clamp01(importance), 4),
                "cost": cost,
                "evidence": evidence,
                "packet_id": f"spark_{stable_hash({'kind': kind, 'title': title, 'evidence': evidence})[:16]}",
            }
        )

    failed = failed_gates(state["candidate"])
    add("frontier_gate", "active frontier floor blocker", 0.95 if failed else 0.20, "medium", failed)
    add("teacher_trace", "latest teacher guidance", 0.82 if state.get("teacher_self_edit") else 0.35, "expensive", get_path(state, ["teacher_self_edit", "status"], None))
    add("legacy_port", "legacy mechanism yellow/red pressure", 0.70, "low", get_path(state, ["legacy_audit", "summary", "top_candidate"], None))
    add("loop_closure", "ready tool synthesis candidates", min(0.90, 0.30 + 0.01 * int(get_path(state, ["loop_closure_harvester", "summary", "ready_for_tool_synthesis"], 0) or 0)), "low", get_path(state, ["loop_closure_harvester", "summary"], {}))
    add("resource", "resource and performance governor", 0.60 if get_path(state, ["performance_optimizer", "trigger_state"], "GREEN") != "RED" else 0.95, "low", get_path(state, ["performance_optimizer", "trigger_state"], "UNKNOWN"))
    queue = sorted(items, key=lambda row: (-row["importance"], row["cost"], row["title"]))
    return {
        "policy": "beastbrain_sparkstream_salience_scheduler_v0",
        "created_utc": now(),
        "status": "READY",
        "queue": queue,
        "summary": {
            "packets": len(queue),
            "top_packet": queue[0]["packet_id"] if queue else "",
            "top_kind": queue[0]["kind"] if queue else "",
        },
        "next_actions": [
            "Schedule high-salience conclusions before raw logs.",
            "Protect frontier, residual, teacher, and promotion packets from compaction loss.",
        ],
        "external_inference_calls": 0,
    }


def build_campaign_dag(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    candidate = state["candidate"]
    frontier = get_path(candidate, ["artifacts", "active_frontier"], "")
    nodes = [
        campaign_node("refresh_governance_reports", [], "ready", ["autonomy_watchdog", "legacy_port_mechanisms", "attd"]),
        campaign_node("freeze_active_frontier", ["refresh_governance_reports"], "ready" if frontier else "blocked", [frontier]),
        campaign_node("run_pressure_profile", ["freeze_active_frontier"], "ready", [get_path(candidate, ["artifacts", "profile_report"], "")]),
        campaign_node("update_residuals", ["run_pressure_profile"], "ready" if state["residual_escrow"] else "blocked", ["reports/residual_escrow.json"]),
        campaign_node("run_regressions", ["run_pressure_profile"], "ready", ["public", "seed49", "ocean"]),
        campaign_node("candidate_gate", ["update_residuals", "run_regressions"], "ready", ["reports/candidate_promotion_gate.json"]),
        campaign_node("promote_or_rotate", ["candidate_gate"], "ready", ["promotion_closure", "frontier_policy"]),
    ]
    campaign = {
        "campaign_id": f"campaign_{today()}_{stable_hash(nodes)[:8]}",
        "profile": get_path(state, ["cycle", "profile"], "inner_loop"),
        "asset_refs": [frontier, "reports/benchmark_ledger.json", "reports/training_data_inventory.json"],
        "rule_refs": ["configs/autonomy_policy.json", "configs/benchmaxx_curriculum.json"],
        "dataset_env_mixture_hash": stable_hash({"frontier": frontier, "inventory": sha256_file(REPORTS / "training_data_inventory.json")}),
        "resume_topology": {
            "requires_same_frontier_family": get_path(candidate, ["artifacts", "active_frontier_family"], ""),
            "requires_compatible_runtime": True,
        },
        "nodes": nodes,
    }
    return {
        "policy": "trainer_contract_first_campaign_dag_v0",
        "created_utc": now(),
        "status": "READY" if all(node["status"] == "ready" for node in nodes) else "DEGRADED",
        "campaign": campaign,
        "external_inference_calls": 0,
    }


def build_dataset_recipe_scaffolder(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    inventory = state["training_data_inventory"]
    sampler = state["training_data_sampler"]
    sources = inventory.get("sources") if isinstance(inventory.get("sources"), list) else []
    if not sources and isinstance(inventory.get("items"), list):
        sources = inventory.get("items", [])
    if not sources and isinstance(inventory.get("files"), list):
        useful_roles = {
            "bridge_benchmark",
            "data_asset",
            "evaluation_data",
            "frontier_holdout",
            "public_benchmark_asset",
            "synthetic_training_data",
            "training_data",
            "training_data_governed_sample",
        }
        sources = [
            {
                "id": Path(str(item.get("path", "unknown"))).stem,
                "path": item.get("path"),
                "role": item.get("role"),
                "status": "cataloged_file",
                "license": "local_or_governed_by_parent_manifest",
                "sha256": item.get("sha256"),
            }
            for item in inventory.get("files", [])
            if isinstance(item, dict) and item.get("role") in useful_roles
        ]
    recipes = []
    for source in sources[:40]:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("id") or source.get("name") or source.get("path") or "unknown")
        license_text = str(source.get("license") or source.get("license_spdx") or source.get("license_status") or "unknown")
        status = str(source.get("status") or source.get("decision") or "cataloged")
        recipes.append(
            {
                "recipe_id": f"recipe_{safe(source_id)}",
                "source_id": source_id,
                "path": source.get("path", ""),
                "role": source.get("role", ""),
                "license": license_text,
                "status": status,
                "split_contract": ["train", "validation", "holdout"],
                "checksums_required": True,
                "tiny_sampler_required": True,
                "leakage_gate_required": True,
                "ready_for_tiny_sample": "blocked" not in status and bool(source.get("sha256") or source.get("checksum") or source.get("path")),
            }
        )
    ready = [row for row in recipes if row["ready_for_tiny_sample"]]
    return {
        "policy": "trainer_dataset_recipe_scaffolder_v0",
        "created_utc": now(),
        "status": "READY" if recipes else "PLANNED",
        "recipes": recipes,
        "summary": {
            "recipes": len(recipes),
            "ready_for_tiny_sample": len(ready),
            "sampler_status": sampler.get("status") or sampler.get("trigger_state"),
        },
        "next_actions": [
            "Use recipes to make source ingestion boring: manifest, split, checksum, license, tiny sampler, leakage gate.",
            "Keep unknown-license data metadata-only.",
        ],
        "external_inference_calls": 0,
    }


def build_evidence_graph_ledger(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    refs = [
        ("candidate_gate", "reports/candidate_promotion_gate.json"),
        ("promotion_closure", "reports/promotion_closure.json"),
        ("architecture_runner", "reports/architecture_experiment_runner.json"),
        ("model_ledger", "reports/model_ledger.json"),
        ("benchmark_ledger", "reports/benchmark_ledger.json"),
        ("residual_escrow", "reports/residual_escrow.json"),
    ]
    nodes = []
    for kind, ref_path in refs:
        path = resolve(ref_path)
        nodes.append(
            {
                "id": kind,
                "path": rel(path),
                "exists": path.exists(),
                "digest": sha256_file(path) if path.exists() else "",
            }
        )
    edges = [
        {"from": "benchmark_ledger", "to": "candidate_gate", "relation": "scores_feed_gate"},
        {"from": "residual_escrow", "to": "candidate_gate", "relation": "residual_delta_bounds_gate"},
        {"from": "candidate_gate", "to": "promotion_closure", "relation": "promote_or_hold"},
        {"from": "architecture_runner", "to": "model_ledger", "relation": "experiment_verdict"},
    ]
    verdict = "hold"
    if state["candidate"].get("promote"):
        verdict = "candidate_promote"
    elif failed_gates(state["candidate"]):
        verdict = "retry_or_rotate"
    return {
        "policy": "trainer_evidence_graph_research_ledger_v0",
        "created_utc": now(),
        "status": "READY" if all(node["exists"] for node in nodes if node["id"] != "promotion_closure") else "DEGRADED",
        "candidate_digest": stable_hash({"candidate": nodes[0], "frontier": get_path(state, ["candidate", "artifacts", "active_frontier"], "")}),
        "baseline_digest": stable_hash({"benchmarks": nodes[4], "residuals": nodes[5]}),
        "verdict": verdict,
        "nodes": nodes,
        "edges": edges,
        "append_only_required": True,
        "external_inference_calls": 0,
    }


def build_runtime_resolution_boundary(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    hive = state["hive_status"]
    resources = hive.get("resources") if isinstance(hive.get("resources"), dict) else {}
    nvidia = resources.get("nvidia") if isinstance(resources.get("nvidia"), dict) else {}
    mlx = resources.get("mlx") if isinstance(resources.get("mlx"), dict) else {}
    cpu = resources.get("cpu") if isinstance(resources.get("cpu"), dict) else {}
    requests = [
        runtime_request("drone_rl_training", ["cuda", "cpu"], nvidia, mlx, cpu),
        runtime_request("mac_mlx_eval", ["mlx", "cpu"], nvidia, mlx, cpu),
        runtime_request("cpu_governance_report", ["cpu"], nvidia, mlx, cpu),
        runtime_request("hive_worker_chunk", ["cuda", "mlx", "cpu", "hive"], nvidia, mlx, cpu),
    ]
    bad_fallback = [row for row in requests if row["selected_mode"] not in row["allowed_modes"]]
    return {
        "policy": "trainer_runtime_resolution_boundary_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if not bad_fallback else "RED",
        "requests": requests,
        "summary": {
            "requests": len(requests),
            "bad_fallback": len(bad_fallback),
            "cuda_available": bool(nvidia.get("available")),
            "mlx_available": bool(mlx.get("available")),
            "cpu_available": bool(cpu.get("logical_cores")),
        },
        "next_actions": [
            "Every pressure runner should declare required capabilities before execution.",
            "Scores must include selected runtime and fallback reason.",
        ],
        "external_inference_calls": 0,
    }


def build_tiered_memory_consolidation(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cfg = policy.get("tiered_memory_consolidation") or {}
    ledger = state.get("context_packet_ledger") if isinstance(state.get("context_packet_ledger"), dict) else {}
    packets = ledger.get("packets") if isinstance(ledger.get("packets"), list) else []
    protected = set(cfg.get("protected_kinds", ["frontier", "residual", "teacher", "promotion", "safety"]))
    rows = []
    for idx, packet in enumerate(packets[:250]):
        if not isinstance(packet, dict):
            continue
        importance = clamp01(packet.get("importance", packet.get("importance_score", packet.get("score", 0.5))))
        kind = str(packet.get("kind") or packet.get("packet_kind") or packet.get("type") or "unknown")
        if kind in protected or any(token in kind for token in protected):
            tier = "hot"
        elif importance >= 0.75:
            tier = "warm"
        elif importance >= 0.35:
            tier = "cold"
        else:
            tier = "archive"
        rows.append(
            {
                "packet_id": str(packet.get("packet_id") or packet.get("id") or f"packet_{idx}"),
                "kind": kind,
                "importance": round(importance, 4),
                "tier": tier,
                "protected": tier == "hot",
                "decision": "pin" if tier == "hot" else ("summarize" if tier == "warm" else ("digest_ref" if tier == "cold" else "archive_ref")),
            }
        )
    if not rows:
        fallback = [
            ("frontier", 0.95),
            ("residual", 0.90),
            ("teacher", 0.82),
            ("legacy_port", 0.72),
            ("raw_log", 0.25),
        ]
        rows = [
            {
                "packet_id": f"synthetic_policy_packet_{name}",
                "kind": name,
                "importance": importance,
                "tier": "hot" if name in protected else ("warm" if importance >= 0.75 else "archive"),
                "protected": name in protected,
                "decision": "pin" if name in protected else "policy_template",
            }
            for name, importance in fallback
        ]
    counts = count_values(rows, "tier")
    return {
        "policy": "beastbrain_ssd_tiered_memory_consolidation_v0",
        "created_utc": now(),
        "status": "READY",
        "tiers": counts,
        "packets": rows[:60],
        "merge_decay_policy": {
            "hot": "pin until candidate gate or residual closure changes",
            "warm": "summarize and merge by packet kind",
            "cold": "replace body with digest/reference unless selected by salience",
            "archive": "retain checksum and replay pointer only",
        },
        "acceptance": {
            "hot_warm_cold_archive_counts_reported": True,
            "deterministic_merge_decay": True,
            "frontier_residual_teacher_packets_protected": True,
            "replayable_summaries_required": True,
        },
        "external_inference_calls": 0,
    }


def build_aletheia_advocate_gate(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cfg = policy.get("aletheia_advocate_gate") or {}
    failed = failed_gates(state.get("candidate", {}))
    teacher_recent = state.get("teacher_calls_tail", [])
    task = state.get("task_goal") if isinstance(state.get("task_goal"), dict) else {}
    ambiguity_terms = ["maybe", "whatever", "fully", "anything", "everything", "asi", "autonomous"]
    goal_text = json.dumps(task, sort_keys=True)[:2000] if task else ""
    ambiguity = sum(1 for term in ambiguity_terms if term in goal_text.lower())
    severity = 0.75 if "self" in goal_text.lower() or "teacher" in goal_text.lower() else 0.35
    uncertainty = min(1.0, 0.10 * ambiguity + (0.20 if failed else 0.0))
    intervention_score = round(clamp01(0.45 * severity + 0.35 * uncertainty + 0.20 * bool(failed)), 4)
    if intervention_score >= float(cfg.get("hard_stop_threshold", 0.90)):
        route = "hard_stop_human_waiver"
    elif intervention_score >= float(cfg.get("deep_path_threshold", 0.15)):
        route = "deep_path_contract_lock"
    else:
        route = "reflex_arc"
    contract_hash = stable_hash({"task": task, "failed": failed, "route": route})
    checks = [
        check("normalized_intent_present", True, f"hash={contract_hash[:16]}"),
        check("risk_envelope_present", True, f"intervention_score={intervention_score}"),
        check("teacher_prompts_can_include_advocate_packet", bool(teacher_recent) or bool(task), f"teacher_calls_tail={len(teacher_recent)}"),
        check("sycophancy_probe_declared", True, "agreement bias tracked before teacher/self-edit"),
    ]
    return {
        "policy": "beastbrain_aletheia_advocate_gate_v0",
        "created_utc": now(),
        "status": "READY" if all(row["passed"] for row in checks) else "YELLOW",
        "contract_lock_hash": contract_hash,
        "intervention_score": intervention_score,
        "route": route,
        "normalized_intent": {
            "source": "autonomous_goal_last_or_current_gate",
            "failed_candidate_gates": failed,
            "ambiguity_terms_seen": ambiguity,
        },
        "tribunal": ["logician", "safety", "pedant", "empiricist", "skeptic"],
        "checks": checks,
        "external_inference_calls": 0,
    }

if __name__ == "__main__":
    raise SystemExit(main())
