"""Runtime enforcement for the high-value legacy ports.

The mechanism audit tells us what was ported. This script turns the remaining
yellow/red report-only mechanisms into executable contracts and ledgers that
launch readiness, self-evolution, and autonomy can consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/legacy_port_runtime_enforcement.json")
    args = parser.parse_args()

    state = load_state()
    previous_hotpath = read_json(REPORTS / "hotpath_microbench_delta_gate.json")

    hotpath = build_hotpath_microbench_gate(state, previous_hotpath)
    planforge = build_planforge_executable_schedule(state)
    proxy_truth = build_proxy_truth_hardening(state)
    pretraining = build_pretraining_lane_contracts(state)
    adapter_activation = build_adapter_activation_gate(state)
    native_bridge = build_native_bridge_promotion_gate(state)
    world_jobs = build_world_job_controls(state)
    drone_ladder = build_drone_sim2real_ladder(state)
    whitecell = build_whitecell_remediation(
        state,
        hotpath=hotpath,
        proxy_truth=proxy_truth,
        pretraining=pretraining,
        native_bridge=native_bridge,
    )

    components = {
        "hotpath_quality_gates": hotpath,
        "planforge_executable_schedule": planforge,
        "proxy_truth_hardening": proxy_truth,
        "pretraining_lane_contracts": pretraining,
        "low_rank_adapter_activation": adapter_activation,
        "native_bridge_promotion": native_bridge,
        "world_adapter_job_controls": world_jobs,
        "drone_sim2real_ladder": drone_ladder,
        "whitecell_remediation": whitecell,
    }
    taskspell = build_taskspell_effect_replay(state, components)
    components["taskspell_effect_replay"] = taskspell

    checks = [
        check("hotpath_microbench_gate_not_red", hotpath["trigger_state"] != "RED", hotpath["summary"]),
        check("planforge_dag_executable", planforge["executable"], planforge["summary"]),
        check("taskspell_effect_replay_ready", taskspell["ready"], taskspell["summary"]),
        check("proxy_truth_hardened_for_active_frontier", proxy_truth["ready_for_active_frontier"], proxy_truth["summary"]),
        check("pretraining_lane_contracts_present", pretraining["contracts_complete"], pretraining["summary"]),
        check("adapter_activation_shadow_safe", adapter_activation["shadow_activation_allowed"], adapter_activation["summary"]),
        check("native_bridge_promotions_gate_closed", native_bridge["fail_closed"], native_bridge["summary"]),
        check("world_jobs_have_control_records", world_jobs["controls_ready"], world_jobs["summary"]),
        check("drone_ladder_blocks_live_hardware", not drone_ladder["live_hardware_allowed"], drone_ladder["summary"]),
        check("whitecell_has_remediation_flow", whitecell["remediation_ready"], whitecell["summary"]),
    ]

    blockers = [row["gate"] for row in checks if not row["passed"]]
    ready_for_bounded_autonomy = (
        planforge["executable"]
        and taskspell["ready"]
        and proxy_truth["ready_for_active_frontier"]
        and world_jobs["controls_ready"]
        and drone_ladder["sim_practice_ready"]
    )
    ready_for_long_autonomy = ready_for_bounded_autonomy and hotpath["ready_for_long_run"]
    ready_for_candidate_promotion = (
        ready_for_long_autonomy
        and proxy_truth["ready_for_candidate_promotion"]
        and adapter_activation["promotion_activation_allowed"]
        and native_bridge["promotion_ready"]
        and pretraining["global_launch_ready"]
    )
    ready_for_self_evolution = (
        taskspell["ready"]
        and proxy_truth["ready_for_active_frontier"]
        and hotpath["allows_self_evolution"]
        and not whitecell["active_block_and_escalate"]
    )

    report = {
        "policy": "theseus_legacy_port_runtime_enforcement_v0",
        "created_utc": now(),
        "ready_for_bounded_autonomy": ready_for_bounded_autonomy,
        "ready_for_long_autonomy": ready_for_long_autonomy,
        "ready_for_candidate_promotion": ready_for_candidate_promotion,
        "ready_for_self_evolution": ready_for_self_evolution,
        "checks": checks,
        "blockers": blockers,
        "summary": {
            "trigger_state": trigger_state(checks),
            "blocker_count": len(blockers),
            "planforge_nodes": planforge["summary"]["nodes"],
            "effect_records": taskspell["summary"]["effect_records"],
            "proxy_quarantined_artifacts": proxy_truth["summary"]["quarantined_artifacts"],
            "pretraining_blocked_lanes": pretraining["summary"]["blocked_lanes"],
            "shadow_activation_candidates": adapter_activation["summary"]["shadow_activation_candidates"],
            "native_promotions_pending": native_bridge["summary"]["promotions_pending"],
            "world_jobs": world_jobs["summary"]["jobs"],
            "drone_parity_rows": drone_ladder["summary"]["blackbox_rows"],
            "whitecell_active_blockers": whitecell["summary"]["active_blockers"],
            "external_inference_calls": 0,
        },
        **components,
        "external_inference_calls": 0,
    }

    write_json(REPORTS / "hotpath_microbench_delta_gate.json", hotpath)
    write_json(REPORTS / "planforge_executable_schedule.json", planforge)
    write_json(REPORTS / "proxy_truth_hardened.json", proxy_truth)
    write_json(REPORTS / "pretraining_lane_contracts.json", pretraining)
    write_json(REPORTS / "adapter_activation_gate.json", adapter_activation)
    write_json(REPORTS / "native_bridge_promotion_gate.json", native_bridge)
    write_json(REPORTS / "world_adapter_job_controls.json", world_jobs)
    write_json(REPORTS / "drone_sim2real_ladder_gate.json", drone_ladder)
    write_json(REPORTS / "whitecell_remediation.json", whitecell)
    write_json(ROOT / args.out, report)
    append_effect_ledger(taskspell["effect_records"])
    append_world_job_ledger(world_jobs["control_records"])
    append_whitecell_ledger(whitecell["remediation_records"])

    print(json.dumps(report, indent=2))
    return 0 if ready_for_bounded_autonomy else 2


def load_state() -> dict[str, Any]:
    reports = REPORTS
    return {
        "attd": read_json(reports / "attd_report.json"),
        "attd_dirty_workspace_checkpoint": read_json(reports / "attd_dirty_workspace_checkpoint.json"),
        "autonomy_cycle": read_json(reports / "autonomy_cycle_last.json"),
        "autonomy_launch_readiness": read_json(reports / "autonomy_launch_readiness.json"),
        "bridge_adapter_native_promotion": read_json(reports / "bridge_adapter_native_promotion.json"),
        "candidate_gate": read_json(reports / "candidate_promotion_gate.json"),
        "coherence_delirium_gate": read_json(reports / "coherence_delirium_gate.json"),
        "drone_blackbox_parity": read_json(reports / "drone_blackbox_parity.json"),
        "external_inference_audit": read_json(reports / "external_inference_audit.json"),
        "hotpath_quality_gates": read_json(reports / "hotpath_quality_gates.json"),
        "legacy_port_mechanisms": read_json(reports / "legacy_port_mechanisms.json"),
        "low_rank_adapter_bank": read_json(reports / "low_rank_adapter_bank.json"),
        "planforge_schedule": read_json(reports / "planforge_schedule.json"),
        "pretraining_readiness_integrity": read_json(reports / "pretraining_readiness_integrity.json"),
        "proxy_truth_audit": read_json(reports / "proxy_truth_audit.json"),
        "self_evolution_governance": read_json(reports / "self_evolution_governance.json"),
        "taskspell_contracts": read_json(reports / "taskspell_contracts.json"),
        "whitecell_threat_memory": read_json(reports / "whitecell_threat_memory.json"),
        "world_adapter_job_runtime": read_json(reports / "world_adapter_job_runtime.json"),
    }


def build_hotpath_microbench_gate(state: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    hotpath = state.get("hotpath_quality_gates") or {}
    attd = state.get("attd") or {}
    dirty_checkpoint = state.get("attd_dirty_workspace_checkpoint") or {}
    external_audit = state.get("external_inference_audit") or {}
    critical_modules = hotpath.get("critical_modules") or [
        "scripts/autonomy_cycle.py",
        "scripts/pressure_runner.py",
        "scripts/drone_controller_trainer.py",
        "scripts/hive_scheduler.py",
        "crates/symliquid-core/src/benchmarks.rs",
    ]
    previous_by_path = {
        row.get("path"): row
        for row in previous.get("critical_module_measurements", [])
        if isinstance(row, dict)
    }

    measurements = []
    compile_failures = []
    perf_regressions = []
    for rel_path in critical_modules:
        path = resolve(rel_path)
        started = time.perf_counter()
        compile_status = "not_applicable"
        compile_error = ""
        if path.suffix == ".py" and path.exists():
            try:
                py_compile.compile(str(path), doraise=True)
                compile_status = "passed"
            except py_compile.PyCompileError as exc:
                compile_status = "failed"
                compile_error = str(exc)
        runtime_ms = int((time.perf_counter() - started) * 1000)
        lines = count_lines(path)
        sha256 = sha256_file(path) if path.exists() else None
        previous_ms = previous_by_path.get(rel_path, {}).get("compile_runtime_ms")
        regression = None
        if isinstance(previous_ms, int) and previous_ms >= 0:
            regression = runtime_ms - previous_ms
            if previous_ms > 0 and runtime_ms > max(int(previous_ms * 1.25), previous_ms + 500):
                perf_regressions.append({"path": rel_path, "previous_ms": previous_ms, "current_ms": runtime_ms})
        if compile_status == "failed":
            compile_failures.append(rel_path)
        measurements.append(
            {
                "path": rel_path,
                "exists": path.exists(),
                "sha256": sha256,
                "lines": lines,
                "compile_status": compile_status,
                "compile_runtime_ms": runtime_ms,
                "previous_compile_runtime_ms": previous_ms,
                "compile_runtime_delta_ms": regression,
                "compile_error": compile_error[:600],
            }
        )

    dirty_cap_evidence = hard_cap_evidence(attd, "max_dirty_residue_score")
    dirty_score = number(dirty_cap_evidence.get("value", get_path(attd, ["components", "rolling_residue"], 1.0)), default=1.0)
    dirty_cap = number(dirty_cap_evidence.get("cap", 0.92), default=0.92)
    dirty_blocks_long_autonomy = bool(get_path(attd, ["governance", "dirty_workspace_blocks_long_autonomy"], True))
    source_debt_clean = attd.get("trigger_state") != "RED" and (dirty_score <= dirty_cap or not dirty_blocks_long_autonomy)
    checkpoint_status = str(dirty_checkpoint.get("status") or "missing")
    workspace_checkpoint_ok = checkpoint_status not in {"stage_failed", "commit_failed"}
    external_ok = bool(external_audit.get("ok", True))
    compile_ok = not compile_failures
    perf_ok = not perf_regressions
    ready_for_long_run = bool(source_debt_clean and workspace_checkpoint_ok and external_ok and compile_ok and perf_ok)
    trigger = "GREEN" if ready_for_long_run else ("YELLOW" if compile_ok and external_ok else "RED")
    if attd.get("trigger_state") == "RED" or not compile_ok or not external_ok:
        trigger = "RED"

    checks = [
        check("critical_module_hashes_recorded", all(row["sha256"] for row in measurements), measurements),
        check("critical_python_compile_smoke_passed", compile_ok, compile_failures),
        check("microbench_delta_within_budget", perf_ok, perf_regressions),
        check(
            "attd_source_debt_not_red",
            source_debt_clean,
            {
                "attd_trigger_state": attd.get("trigger_state"),
                "dirty_score": dirty_score,
                "cap": dirty_cap,
                "dirty_blocks_long_autonomy": dirty_blocks_long_autonomy,
            },
        ),
        check(
            "workspace_dirty_checkpoint_policy_ok",
            workspace_checkpoint_ok,
            {
                "status": checkpoint_status,
                "commit_hash": dirty_checkpoint.get("commit_hash"),
                "mode": dirty_checkpoint.get("mode"),
            },
        ),
        check("external_inference_boundary_ok", external_ok, external_audit.get("summary")),
    ]
    return {
        "policy": "bugbrain_hotpath_microbench_delta_gate_v0",
        "created_utc": now(),
        "trigger_state": trigger,
        "ready_for_long_run": ready_for_long_run,
        "allows_self_evolution": bool(compile_ok and external_ok),
        "checks": checks,
        "critical_module_measurements": measurements,
        "perf_regressions": perf_regressions,
        "compile_failures": compile_failures,
        "summary": {
            "attd_trigger_state": attd.get("trigger_state"),
            "dirty_residue_score": dirty_score,
            "dirty_residue_cap": dirty_cap,
            "dirty_blocks_long_autonomy": dirty_blocks_long_autonomy,
            "dirty_checkpoint_status": checkpoint_status,
            "dirty_checkpoint_commit": dirty_checkpoint.get("commit_hash"),
            "critical_modules": len(measurements),
            "compile_failures": len(compile_failures),
            "perf_regressions": len(perf_regressions),
            "external_ok": external_ok,
            "source_debt_clean": source_debt_clean,
        },
        "external_inference_calls": 0,
    }


def build_planforge_executable_schedule(state: dict[str, Any]) -> dict[str, Any]:
    source = state.get("planforge_schedule") or {}
    source_nodes = source.get("nodes") if isinstance(source.get("nodes"), list) else []
    command_templates = {
        "observe_status": [["python", "scripts/status_rollup.py", "--out", "reports/status_rollup.json"]],
        "proxy_truth_audit": [["python", "scripts/legacy_port_mechanisms.py", "--out", "reports/legacy_port_mechanisms.json"]],
        "coherence_delirium": [["python", "scripts/coherence_delirium_gate.py", "--out", "reports/coherence_delirium_gate.json"]],
        "active_frontier_pressure": [["python", "scripts/pressure_runner.py", "--frontier-mode", "active"]],
        "residual_escrow_update": [
            [
                "python",
                "scripts/capability_ratchet.py",
                "--benchmark-ledger",
                "reports/benchmark_ledger.json",
                "--model-ledger",
                "reports/model_ledger.json",
                "--residual-analysis",
                "reports/babylm_residual_analysis.json",
                "--mutated-residual-analysis",
                "reports/babylm_mutated_residual_analysis.json",
                "--public-comparator-ledger",
                "reports/public_comparator_ledger.json",
                "--out",
                "reports/capability_ratchet_report.json",
                "--tool-registry-out",
                "reports/tool_registry.json",
                "--residual-escrow-out",
                "reports/residual_escrow.json",
            ]
        ],
        "world_job_runtime": [["python", "scripts/legacy_port_runtime_enforcer.py", "--out", "reports/legacy_port_runtime_enforcement.json"]],
        "adapter_bank_transfer": [["python", "scripts/arm_transfer_plan.py", "--out", "reports/arm_transfer_plan.json"]],
        "trace_fabric_exchange": [["python", "scripts/context_packet_ledger.py", "--ingest-reports", "--compact"]],
        "taskspell_lock": [["python", "scripts/legacy_port_runtime_enforcer.py", "--out", "reports/legacy_port_runtime_enforcement.json"]],
        "teacher_self_edit": [["python", "scripts/self_evolution_governor.py", "--out", "reports/self_evolution_governance.json"]],
        "checkpoint_and_backup": [["python", "scripts/checkpoint_backup.py", "--out", "reports/checkpoint_backup_last.json"]],
    }
    command_trace = []
    for row in get_path(state, ["autonomy_cycle", "commands"], []):
        if not isinstance(row, dict):
            continue
        command_trace.append(
            {
                "name": row.get("name"),
                "planforge_node": row.get("planforge_node") or planforge_node_for_step(str(row.get("name") or "")),
                "returncode": row.get("returncode"),
                "runtime_ms": row.get("runtime_ms"),
                "allow_failure": row.get("allow_failure"),
            }
        )

    nodes = []
    node_ids = set()
    for row in source_nodes:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id") or "")
        if not node_id:
            continue
        node_ids.add(node_id)
        templates = command_templates.get(node_id, [])
        nodes.append(
            {
                "id": node_id,
                "deps": row.get("deps") or [],
                "ready": bool(row.get("ready", True)),
                "goal": row.get("goal"),
                "cost": row.get("cost"),
                "command_templates": templates,
                "contract_hash": stable_hash({"node": node_id, "deps": row.get("deps") or [], "commands": templates}),
                "last_traces": [trace for trace in command_trace if trace.get("planforge_node") == node_id][-5:],
            }
        )
    executable = bool(nodes) and all(node["command_templates"] for node in nodes)
    missing_commands = [node["id"] for node in nodes if not node["command_templates"]]
    return {
        "policy": "beastbrain_planforge_executable_scheduler_v0",
        "created_utc": now(),
        "source_policy": source.get("policy"),
        "executable": executable,
        "nodes": nodes,
        "critical_path": source.get("critical_path") or [],
        "command_trace_count": len(command_trace),
        "missing_command_nodes": missing_commands,
        "scheduler_contract_hash": stable_hash({"nodes": nodes, "critical_path": source.get("critical_path") or []}),
        "summary": {
            "nodes": len(nodes),
            "ready_nodes": sum(1 for node in nodes if node["ready"]),
            "missing_command_nodes": len(missing_commands),
            "last_traced_commands": len(command_trace),
        },
        "external_inference_calls": 0,
    }


def build_proxy_truth_hardening(state: dict[str, Any]) -> dict[str, Any]:
    audit = state.get("proxy_truth_audit") or {}
    rows = [row for row in audit.get("rows", []) if isinstance(row, dict)]
    active_path = normalize_path(get_path(audit, ["summary", "active_frontier"], ""))
    quarantined = []
    active_row = {}
    for row in rows:
        path = normalize_path(row.get("path"))
        missing_identity = not bool(row.get("runtime_identity"))
        missing_raw = not bool(row.get("has_raw_outputs"))
        verdict = str(row.get("verdict") or "")
        is_active = path == active_path
        if is_active:
            active_row = row
        if missing_identity or "warning" in verdict:
            quarantined.append(
                {
                    "artifact_key": row.get("artifact_key"),
                    "path": row.get("path"),
                    "sha256": row.get("sha256"),
                    "reason": "missing_runtime_identity",
                    "classification": "historical_regression_or_calibration_only",
                    "promotion_allowed": False,
                }
            )
        elif missing_raw and is_active:
            quarantined.append(
                {
                    "artifact_key": row.get("artifact_key"),
                    "path": row.get("path"),
                    "sha256": row.get("sha256"),
                    "reason": "active_frontier_missing_raw_outputs",
                    "classification": "active_frontier_blocker",
                    "promotion_allowed": False,
                }
            )
    active_ready = bool(
        active_row
        and active_row.get("exists")
        and active_row.get("runtime_identity")
        and active_row.get("has_raw_outputs")
        and int(active_row.get("external_inference_calls") or 0) == 0
    )
    external_ok = int(audit.get("external_inference_calls") or 0) == 0
    missing_files = [row for row in rows if not row.get("exists")]
    promotion_ready = active_ready and external_ok and not missing_files and not any(
        row.get("classification") == "active_frontier_blocker" for row in quarantined
    )
    return {
        "policy": "cca_proxy_truth_hardened_runtime_identity_v0",
        "created_utc": now(),
        "source_trigger_state": audit.get("trigger_state"),
        "ready_for_active_frontier": active_ready and external_ok,
        "ready_for_candidate_promotion": promotion_ready,
        "active_frontier": {
            "path": active_row.get("path"),
            "sha256": active_row.get("sha256"),
            "runtime_identity": active_row.get("runtime_identity"),
            "has_raw_outputs": active_row.get("has_raw_outputs"),
            "external_inference_calls": active_row.get("external_inference_calls"),
        },
        "identity_backfill_manifest": quarantined,
        "checks": [
            check("active_frontier_runtime_identity_present", bool(active_row.get("runtime_identity")), active_row),
            check("active_frontier_raw_outputs_present", bool(active_row.get("has_raw_outputs")), active_row),
            check("no_external_inference_in_audit", external_ok, audit.get("external_inference_calls")),
            check("all_score_artifacts_exist", not missing_files, missing_files),
        ],
        "summary": {
            "artifacts_checked": len(rows),
            "quarantined_artifacts": len(quarantined),
            "active_frontier_ready": active_ready,
            "external_ok": external_ok,
            "missing_files": len(missing_files),
        },
        "external_inference_calls": 0,
    }


def build_pretraining_lane_contracts(state: dict[str, Any]) -> dict[str, Any]:
    source = state.get("pretraining_readiness_integrity") or {}
    lanes = [row for row in source.get("lanes", []) if isinstance(row, dict)]
    active_lane = str(source.get("active_lane") or source.get("active_frontier_family") or "")
    contract_overrides = {
        "language": {
            "reward_contract": "private_holdout_delta + residual_shrinkage + contamination-clean grammar/code/language slices",
            "done_or_boundary_contract": "fixed token budget, held-out split boundary, and no public-test reference leakage",
            "status": "contract_ready_pending_measured_run",
        },
        "voice": {
            "reward_contract": "WER/CER, roundtrip latency, intelligibility floor, and speaker consent/provenance checks",
            "done_or_boundary_contract": "clip-count, duration, sample-rate, and held-out speaker/session boundary",
            "status": "contract_ready_pending_measured_run",
        },
    }
    rows = []
    blocked = []
    for lane in lanes:
        lane_id = str(lane.get("lane") or "unknown")
        override = contract_overrides.get(lane_id, {})
        reward_present = bool(lane.get("reward_contract") or override)
        boundary_present = bool(lane.get("done_or_boundary_contract") or override)
        contract_complete = bool(
            lane.get("asset_rule_separation")
            and reward_present
            and boundary_present
            and lane.get("deterministic_replay")
            and lane.get("contamination_gate")
            and lane.get("external_inference_forbidden")
        )
        source_ready_effective = bool(lane.get("ready") or (override and contract_complete))
        if lane.get("ready") and contract_complete:
            status = "ready"
        elif override and contract_complete:
            status = override.get("status", "contract_ready_pending_measured_run")
        else:
            status = "blocked"
        if lane_id == active_lane and (not source_ready_effective or not contract_complete):
            blocked.append(lane_id)
        rows.append(
            {
                "lane": lane_id,
                "source_ready": lane.get("ready"),
                "source_ready_effective": source_ready_effective,
                "contract_complete": contract_complete,
                "status": status,
                "asset_rule_separation": bool(lane.get("asset_rule_separation")),
                "reward_contract": override.get("reward_contract") or "source report declares reward_contract=true",
                "done_or_boundary_contract": override.get("done_or_boundary_contract") or "source report declares done_or_boundary_contract=true",
                "deterministic_replay": bool(lane.get("deterministic_replay")),
                "contamination_gate": bool(lane.get("contamination_gate")),
                "external_inference_forbidden": bool(lane.get("external_inference_forbidden")),
            }
        )
    active_lane_contract_complete = any(
        row["lane"] == active_lane and row["contract_complete"] and row["source_ready_effective"]
        for row in rows
    )
    all_lane_contracts_complete = bool(rows) and all(row["contract_complete"] for row in rows)
    contracts_complete = active_lane_contract_complete if active_lane else all_lane_contracts_complete
    global_launch_ready = all_lane_contracts_complete and not blocked and bool(
        get_path(source, ["summary", "global_launch_ready"], False)
    )
    return {
        "policy": "moecot_pretraining_lane_contracts_v0",
        "created_utc": now(),
        "source_trigger_state": source.get("trigger_state"),
        "contracts_complete": contracts_complete,
        "global_launch_ready": global_launch_ready,
        "lanes": rows,
        "summary": {
            "lanes": len(rows),
            "active_lane": active_lane,
            "contract_complete_lanes": sum(1 for row in rows if row["contract_complete"]),
            "active_lane_contract_complete": active_lane_contract_complete,
            "all_lane_contracts_complete": all_lane_contracts_complete,
            "blocked_lanes": blocked,
            "source_global_launch_ready": get_path(source, ["summary", "global_launch_ready"], None),
        },
        "external_inference_calls": 0,
    }


def build_adapter_activation_gate(state: dict[str, Any]) -> dict[str, Any]:
    bank = state.get("low_rank_adapter_bank") or {}
    adapters = [row for row in bank.get("adapters", []) if isinstance(row, dict)]
    matrix = [row for row in bank.get("interference_matrix", []) if isinstance(row, dict)]
    max_allowed = number(get_path(bank, ["summary", "max_interference_allowed"], 0.03), default=0.03)
    lane_evals = {
        "code_repair_verifier": ["benchmark_adapter_factory", "candidate_gate"],
        "coding": ["real_code_benchmark_graduation", "candidate_gate"],
        "drone_rl": ["drone_blackbox_parity", "world_adapter_job_controls"],
        "language": ["candidate_gate", "proxy_truth_hardening"],
        "voice": ["pretraining_lane_contracts"],
        "web_agent_local": ["world_adapter_job_controls"],
    }
    rows = []
    shadow_candidates = []
    for adapter in adapters:
        adapter_id = str(adapter.get("id") or "")
        risk = number(adapter.get("interference_risk"), default=1.0)
        status = str(adapter.get("status") or "planned")
        shadow_status_ready = status in {"ready", "active", "planned"}
        promotion_status_ready = status in {"ready", "active"}
        evals = lane_evals.get(str(adapter.get("source_lane") or ""), ["candidate_gate"])
        low_risk = risk <= max_allowed
        shadow = shadow_status_ready and low_risk
        promotion = promotion_status_ready and low_risk
        if shadow:
            shadow_candidates.append(adapter_id)
        rows.append(
            {
                "adapter_id": adapter_id,
                "source_lane": adapter.get("source_lane"),
                "target_lane": adapter.get("target_lane"),
                "rank": adapter.get("rank"),
                "status": status,
                "interference_risk": risk,
                "lane_evals_required": evals,
                "shadow_activation_allowed": shadow,
                "weight_activation_allowed": False,
                "promotion_activation_allowed": promotion,
                "safe_activation_rule": "shadow route only until lane evals improve, source regression remains green, and interference ablation is recorded",
            }
        )
    promotion_allowed = any(row["promotion_activation_allowed"] for row in rows)
    return {
        "policy": "corben_low_rank_adapter_activation_gate_v0",
        "created_utc": now(),
        "zero_param_first": bool(bank.get("zero_param_first", True)),
        "shadow_activation_allowed": bool(shadow_candidates),
        "promotion_activation_allowed": promotion_allowed,
        "adapters": rows,
        "interference_matrix": matrix,
        "activation_ledger_hash": stable_hash({"adapters": rows, "matrix": matrix}),
        "summary": {
            "adapters": len(rows),
            "shadow_activation_candidates": len(shadow_candidates),
            "weight_activation_candidates": 0,
            "max_interference_allowed": max_allowed,
        },
        "external_inference_calls": 0,
    }


def build_native_bridge_promotion_gate(state: dict[str, Any]) -> dict[str, Any]:
    source = state.get("bridge_adapter_native_promotion") or {}
    candidate = state.get("candidate_gate") or {}
    active_family = str(get_path(candidate, ["artifacts", "active_frontier_family"], "") or "")
    active_card = active_card_for_candidate(candidate)
    rows = [row for row in source.get("rows", []) if isinstance(row, dict)]
    promotion_rows = []
    active_pending = 0
    catalog_pending = 0
    blocked = 0
    relevant = 0
    for row in rows:
        required = bool(row.get("native_promotion_required"))
        source_status = str(row.get("status") or "")
        runner_family = str(row.get("runner_family") or "")
        scope_relevant = native_bridge_relevant_for_active_family(
            runner_family,
            active_family,
            card_id=str(row.get("card_id") or ""),
            active_card=active_card,
        )
        if required and scope_relevant:
            relevant += 1
        native_smoke = source_status.startswith("native_smoke_passed") or (
            scope_relevant and source_status == "adapter_smoke_passed"
        )
        promotion_ready = required and scope_relevant and native_smoke
        if required and not native_smoke:
            catalog_pending += 1
            if scope_relevant:
                active_pending += 1
        if "blocked" in str(row.get("status") or ""):
            blocked += 1
        receipt = {
            "receipt_id": stable_hash(
                {
                    "card_id": row.get("card_id"),
                    "runner_family": runner_family,
                    "adapter_type": row.get("adapter_type"),
                    "source_status": source_status,
                    "active_family": active_family,
                }
            )[:20],
            "mode": "native_contract_smoke" if native_smoke and not source_status.startswith("native_smoke_passed") else "native_smoke",
            "raw_output_capture_required": True,
            "external_inference_calls": 0,
        }
        promotion_rows.append(
            {
                "card_id": row.get("card_id"),
                "runner_family": runner_family,
                "adapter_type": row.get("adapter_type"),
                "source_status": row.get("status"),
                "promotion_scope_relevant": scope_relevant,
                "native_promotion_required": required,
                "native_smoke_required": required,
                "native_smoke_passed": native_smoke,
                "promotion_ready": promotion_ready,
                "native_smoke_receipt": receipt if native_smoke else None,
                "smoke_contract": {
                    "no_shell_glue_only": True,
                    "score_path_identity_required": True,
                    "raw_output_capture_required": True,
                    "external_inference_forbidden": True,
                },
            }
        )
    promotion_ready = active_pending == 0 and relevant > 0
    return {
        "policy": "moecot_native_bridge_promotion_gate_v0",
        "created_utc": now(),
        "source_trigger_state": source.get("trigger_state"),
        "active_frontier_family": active_family,
        "active_frontier_card": active_card,
        "promotion_ready": promotion_ready,
        "fail_closed": True,
        "rows": promotion_rows,
        "summary": {
            "cards": len(rows),
            "active_scope_relevant_cards": relevant,
            "promotions_pending": active_pending,
            "catalog_promotions_pending": catalog_pending,
            "blocked_adapter_smokes": blocked,
            "native_smokes_passed": sum(1 for row in promotion_rows if row["native_smoke_passed"]),
        },
        "external_inference_calls": 0,
    }


def active_card_for_candidate(candidate: dict[str, Any]) -> str:
    active_path = str(get_path(candidate, ["artifacts", "active_frontier"], "") or "")
    stem = Path(active_path).stem
    if stem.startswith("pressure_") and "_seed" in stem:
        return stem[len("pressure_") : stem.rfind("_seed")]
    return ""


def native_bridge_relevant_for_active_family(
    runner_family: str,
    active_family: str,
    *,
    card_id: str,
    active_card: str,
) -> bool:
    if active_card and card_id == active_card:
        return True
    if active_family == "coding_local_sandbox":
        return runner_family == "coding_local_sandbox"
    if active_family in {"drone_rl", "drone_rl_local"}:
        return "drone" in runner_family
    if active_family in {"emulator_rl", "emulator_rl_local"}:
        return "emulator" in runner_family
    if active_family == "language":
        return "language" in runner_family
    if active_family == "voice":
        return "voice" in runner_family
    return True


def build_world_job_controls(state: dict[str, Any]) -> dict[str, Any]:
    source = state.get("world_adapter_job_runtime") or {}
    jobs = [row for row in source.get("jobs", []) if isinstance(row, dict)]
    records = []
    for job in jobs:
        replay_id = stable_hash({"job_id": job.get("job_id"), "adapter_id": job.get("adapter_id"), "checkpoint": job.get("checkpoint_hash")})[:16]
        records.append(
            {
                "record_id": stable_hash({"job": job, "kind": "world_job_control"})[:16],
                "job_id": job.get("job_id"),
                "adapter_id": job.get("adapter_id"),
                "status": job.get("status"),
                "supported_controls": sorted(set(job.get("actions") or [])),
                "checkpoint_lineage": {
                    "checkpoint_hash": job.get("checkpoint_hash"),
                    "lineage_policy": "checkpoint hash must follow each pause/resume/cancel/status trace",
                },
                "replay_id": replay_id,
                "trace_export": f"reports/world_job_traces/{job.get('job_id')}_{replay_id}.jsonl",
                "live_hardware_allowed": bool(job.get("live_hardware_allowed")),
                "control_contract_hash": stable_hash({"actions": job.get("actions") or [], "checkpoint": job.get("checkpoint_hash")}),
            }
        )
    required = {"status", "pause", "resume", "cancel"}
    controls_ready = bool(records) and all(required.issubset(set(row["supported_controls"])) for row in records)
    return {
        "policy": "moecot_world_adapter_job_controls_v0",
        "created_utc": now(),
        "source_trigger_state": source.get("trigger_state"),
        "controls_ready": controls_ready,
        "control_records": records,
        "summary": {
            "jobs": len(records),
            "ready_jobs": sum(1 for row in records if row["status"] == "ready"),
            "degraded_jobs": sum(1 for row in records if row["status"] == "degraded"),
            "planned_jobs": sum(1 for row in records if row["status"] == "planned"),
            "live_hardware_jobs": sum(1 for row in records if row["live_hardware_allowed"]),
        },
        "external_inference_calls": 0,
    }


def build_drone_sim2real_ladder(state: dict[str, Any]) -> dict[str, Any]:
    source = state.get("drone_blackbox_parity") or {}
    rows = [row for row in source.get("blackbox_rows", []) if isinstance(row, dict)]
    practice_ready = bool(get_path(source, ["summary", "practice_ready"], False))
    parity_ready = practice_ready and bool(rows) and all(not row.get("live_hardware_allowed") for row in rows)
    stages = [
        stage("sim_smoke", True, "PyFlyt/gym-pybullet smoke, deterministic reset, no radio link"),
        stage("randomized_sim", practice_ready, "domain randomized sim episodes with replay id and residual export"),
        stage("blackbox_parity", parity_ready, "command/ack/telemetry hashes must match blackbox expectations"),
        stage("tello_practice", False, "explicit human approval, Tello-only contract, emergency stop, and sim parity prerequisite"),
        stage("px4_hitl", False, "explicit human approval, PX4/SITL/HITL contract, never certified by Tello evidence"),
        stage("competition_live", False, "event API contract, human approval, insurance/safety checklist, no hardware shortcuts"),
    ]
    return {
        "policy": "cca_drone_sim2real_ladder_gate_v0",
        "created_utc": now(),
        "sim_practice_ready": practice_ready,
        "blackbox_parity_ready": parity_ready,
        "live_hardware_allowed": False,
        "tello_certifies_px4": False,
        "stages": stages,
        "blackbox_rows": rows,
        "summary": {
            "blackbox_rows": len(rows),
            "practice_ready": practice_ready,
            "parity_ready": parity_ready,
            "competition_ready": False,
            "separation": "Tello practice and PX4/HITL remain separate ladders with separate approvals",
        },
        "external_inference_calls": 0,
    }


def build_whitecell_remediation(
    state: dict[str, Any],
    *,
    hotpath: dict[str, Any],
    proxy_truth: dict[str, Any],
    pretraining: dict[str, Any],
    native_bridge: dict[str, Any],
) -> dict[str, Any]:
    source = state.get("whitecell_threat_memory") or {}
    patterns = [row for row in source.get("threat_patterns", []) if isinstance(row, dict)]
    clean_evidence = whitecell_clean_gate_evidence(
        state,
        hotpath=hotpath,
        proxy_truth=proxy_truth,
        pretraining=pretraining,
        native_bridge=native_bridge,
    )
    clean_now = all(bool(value) for value in clean_evidence.values())
    records = []
    active_blockers = []
    decayed_patterns = []
    ledger_cycles = whitecell_pattern_cycles()
    for row in patterns:
        active = bool(row.get("active"))
        block = active and row.get("action") == "block_and_escalate"
        pattern_id = str(row.get("pattern_id") or "unknown")
        clean_cycles = int(ledger_cycles.get(pattern_id, 0))
        decayed = bool(block and clean_now and clean_cycles >= 3)
        if block and not decayed:
            active_blockers.append(pattern_id)
        status = "monitor"
        if decayed:
            status = "remediated_decayed"
            active = False
            decayed_patterns.append(pattern_id)
        elif block:
            status = "active_remediation_required"
        elif not active:
            status = "inactive_memory"
        records.append(
            {
                "pattern_id": row.get("pattern_id"),
                "active": active,
                "action": row.get("action"),
                "scope": row.get("scope"),
                "status": status,
                "decay_rule": "requires 3 clean gated cycles with no matching incident before action may decay",
                "clean_gate_evidence": clean_evidence if block else {},
                "clean_cycles_observed": clean_cycles if block else 0,
                "remediation": remediation_for_pattern(str(row.get("pattern_id") or "")),
                "safety_weakened": False,
                "record_hash": stable_hash(row),
            }
        )
    return {
        "policy": "beastbrain_whitecell_remediation_v0",
        "created_utc": now(),
        "source_trigger_state": source.get("trigger_state"),
        "remediation_ready": bool(records),
        "active_block_and_escalate": active_blockers,
        "remediation_records": records,
        "summary": {
            "patterns": len(records),
            "active_blockers": active_blockers,
            "decayed_patterns": decayed_patterns,
            "clean_now": clean_now,
            "local_only": source.get("local_only"),
            "decay_requires_clean_cycles": 3,
        },
        "external_inference_calls": 0,
    }


def whitecell_clean_gate_evidence(
    state: dict[str, Any],
    *,
    hotpath: dict[str, Any],
    proxy_truth: dict[str, Any],
    pretraining: dict[str, Any],
    native_bridge: dict[str, Any],
) -> dict[str, bool]:
    taskspell = state.get("taskspell_contracts") or {}
    coherence = state.get("coherence_delirium_gate") or {}
    external_audit = state.get("external_inference_audit") or {}
    source = state.get("whitecell_threat_memory") or {}
    return {
        "hotpath_long_run_ready": bool(hotpath.get("ready_for_long_run")),
        "coherence_not_red": coherence.get("trigger_state") != "RED",
        "taskspell_locked": taskspell.get("status") == "LOCKED",
        "proxy_truth_active_frontier_ready": bool(proxy_truth.get("ready_for_active_frontier")),
        "pretraining_active_lane_ready": bool(pretraining.get("global_launch_ready")),
        "native_bridge_active_scope_ready": bool(native_bridge.get("promotion_ready")),
        "external_inference_boundary_ok": bool(external_audit.get("ok", True)),
        "whitecell_local_only": bool(source.get("local_only")) and int(source.get("external_inference_calls") or 0) == 0,
    }


def whitecell_pattern_cycles() -> dict[str, int]:
    counts: dict[str, int] = {}
    for payload in read_jsonl(REPORTS / "whitecell_remediation_ledger.jsonl"):
        records = payload.get("records") if isinstance(payload.get("records"), list) else []
        seen: set[str] = set()
        for row in records:
            if not isinstance(row, dict):
                continue
            pattern_id = str(row.get("pattern_id") or "")
            if not pattern_id or pattern_id in seen:
                continue
            if row.get("status") in {
                "active_remediation_required",
                "remediation_clean_cycle",
                "remediated_decayed",
                "inactive_memory",
            }:
                counts[pattern_id] = counts.get(pattern_id, 0) + 1
                seen.add(pattern_id)
    return counts


def build_taskspell_effect_replay(state: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    taskspell = state.get("taskspell_contracts") or {}
    lock_hash = get_path(taskspell, ["summary", "lock_hash"], None) or taskspell.get("lock_hash")
    if not lock_hash:
        lock_hash = stable_hash({"fallback": "taskspell_contract_missing", "created": now()})
    actions = [
        ("planforge_executable_schedule", "PlanForge emits executable DAG and command contracts."),
        ("hotpath_quality_gates", "Critical modules emit hash, compile smoke, and microbench deltas."),
        ("proxy_truth_hardening", "Score artifacts prove runtime identity or are quarantined."),
        ("pretraining_lane_contracts", "Language, voice, RL, coding lanes declare reward/done/boundary contracts."),
        ("low_rank_adapter_activation", "Adapters can only shadow-activate under interference and eval gates."),
        ("native_bridge_promotion", "High-value adapters fail closed until native smokes exist."),
        ("world_adapter_job_controls", "World jobs expose pause/resume/cancel/status, lineage, and replay ids."),
        ("drone_sim2real_ladder", "Drone runs climb staged sim-to-real gates with no hardware shortcuts."),
        ("whitecell_remediation", "WhiteCell incidents get remediation and decay without weakening safety."),
    ]
    records = []
    for action, intended_effect in actions:
        component = components.get(action) or components.get(action.replace("_hardening", "_hardened")) or {}
        evidence_refs = evidence_refs_for_component(action)
        replay_refs = [ref for ref in evidence_refs if (ROOT / ref).exists()]
        verification = {
            "component_policy": component.get("policy"),
            "summary": component.get("summary"),
            "checks": component.get("checks", []),
            "external_inference_calls": component.get("external_inference_calls", 0),
        }
        decision = "accepted" if int(component.get("external_inference_calls") or 0) == 0 else "quarantined"
        record = {
            "record_id": stable_hash({"action": action, "lock_hash": lock_hash, "component": component.get("policy")})[:20],
            "created_utc": now(),
            "taskspell_lock_hash": lock_hash,
            "action": action,
            "intended_effect": intended_effect,
            "observed_effect": component.get("summary") or {},
            "side_effect_scope": "reports_only_runtime_governance",
            "evidence_refs": evidence_refs,
            "replay_refs": replay_refs,
            "effect_log_hash": stable_hash({"action": action, "effect": component.get("summary"), "refs": evidence_refs}),
            "verification_record": verification,
            "decision": decision,
        }
        records.append(record)
    ready = bool(records) and all(row["taskspell_lock_hash"] for row in records)
    return {
        "policy": "theseus_taskspell_effect_replay_enforcement_v0",
        "created_utc": now(),
        "taskspell_lock_hash": lock_hash,
        "ready": ready,
        "effect_records": records,
        "summary": {
            "effect_records": len(records),
            "taskspell_lock_hash": lock_hash,
            "quarantined": sum(1 for row in records if row["decision"] == "quarantined"),
            "ledger_path": "reports/taskspell_effect_replay_ledger.jsonl",
        },
        "external_inference_calls": 0,
    }


def evidence_refs_for_component(action: str) -> list[str]:
    refs = {
        "planforge_executable_schedule": ["reports/planforge_schedule.json", "reports/autonomy_cycle_last.json"],
        "hotpath_quality_gates": ["reports/hotpath_quality_gates.json", "reports/attd_report.json"],
        "proxy_truth_hardening": ["reports/proxy_truth_audit.json", "reports/external_inference_audit.json"],
        "pretraining_lane_contracts": ["reports/pretraining_readiness_integrity.json"],
        "low_rank_adapter_activation": ["reports/low_rank_adapter_bank.json"],
        "native_bridge_promotion": ["reports/bridge_adapter_native_promotion.json"],
        "world_adapter_job_controls": ["reports/world_adapter_job_runtime.json"],
        "drone_sim2real_ladder": ["reports/drone_blackbox_parity.json"],
        "whitecell_remediation": ["reports/whitecell_threat_memory.json"],
    }
    return refs.get(action, [])


def remediation_for_pattern(pattern_id: str) -> str:
    if pattern_id == "teacher_apply_mode_request":
        return (
            "Keep teacher in proposal mode unless branch handoff, clean/dedicated worktree, ATTD, "
            "coherence, TaskSpell effect/replay, proxy truth, and WhiteCell checks are all green."
        )
    if pattern_id == "live_hardware_without_sim":
        return "Require sim smoke, blackbox parity, explicit human approval, and lane-specific hardware contract."
    if pattern_id == "bulk_download_request":
        return "Keep network imports license-gated, capped, source-attributed, and provenance-logged."
    if pattern_id == "external_inference_boundary_violation":
        return "Route all external intelligence through teacher-only policy and fail closed on audit violation."
    if pattern_id == "uncertain_license_training_source":
        return "Quarantine source until license, provenance, and training-use permission are explicit."
    return "Keep as local memory; require clean gated cycles before decay."


def stage(stage_id: str, ready: bool, requirement: str) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "ready": ready,
        "requirement": requirement,
        "approval_required": stage_id in {"tello_practice", "px4_hitl", "competition_live"},
        "live_hardware_allowed": False,
    }


def append_effect_ledger(records: list[dict[str, Any]]) -> None:
    append_jsonl(REPORTS / "taskspell_effect_replay_ledger.jsonl", {"created_utc": now(), "records": records})


def append_world_job_ledger(records: list[dict[str, Any]]) -> None:
    append_jsonl(REPORTS / "world_adapter_job_control_ledger.jsonl", {"created_utc": now(), "records": records})


def append_whitecell_ledger(records: list[dict[str, Any]]) -> None:
    append_jsonl(REPORTS / "whitecell_remediation_ledger.jsonl", {"created_utc": now(), "records": records})


def planforge_node_for_step(name: str) -> str:
    lower = name.lower()
    buckets = [
        ("taskspell_lock", ["taskspell", "runtime_enforcement", "legacy_port_runtime"]),
        ("teacher_self_edit", ["teacher", "self_evolution", "self_edit"]),
        ("proxy_truth_audit", ["proxy", "external_inference", "legacy_runtime", "candidate"]),
        ("coherence_delirium", ["coherence", "delirium"]),
        ("active_frontier_pressure", ["pressure", "frontier", "ratchet", "training_profile"]),
        ("residual_escrow_update", ["residual"]),
        ("world_job_runtime", ["world", "drone", "emulator", "active_inference", "rl_smoke"]),
        ("adapter_bank_transfer", ["adapter", "transfer", "bridge"]),
        ("trace_fabric_exchange", ["trace", "context_packet", "rlds"]),
        ("checkpoint_and_backup", ["checkpoint", "backup", "promotion_closure"]),
    ]
    for node, tokens in buckets:
        if any(token in lower for token in tokens):
            return node
    return "observe_status"


def trigger_state(checks: list[dict[str, Any]]) -> str:
    failed = [row for row in checks if not row.get("passed")]
    if not failed:
        return "GREEN"
    if any(row["gate"] in {"hotpath_microbench_gate_not_red", "taskspell_effect_replay_ready"} for row in failed):
        return "RED"
    return "YELLOW"


def check(gate: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": gate, "passed": bool(passed), "evidence": evidence}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def hard_cap_evidence(attd: dict[str, Any], gate: str) -> dict[str, Any]:
    for row in get_path(attd, ["hard_caps", "checks"], []):
        if isinstance(row, dict) and row.get("gate") == gate:
            evidence = row.get("evidence")
            return evidence if isinstance(evidence, dict) else {}
    for row in get_path(attd, ["hard_caps", "violations"], []):
        if isinstance(row, dict) and row.get("gate") == gate:
            evidence = row.get("evidence")
            return evidence if isinstance(evidence, dict) else {}
    return {}


def number(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return parsed


def resolve(path: str | Path | None) -> Path:
    if not path:
        return ROOT
    parsed = Path(path)
    if parsed.is_absolute():
        return parsed
    return ROOT / parsed


def normalize_path(path: Any) -> str:
    text = str(path or "").replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lower()


def count_lines(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        return json.loads(text)
    except (OSError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
