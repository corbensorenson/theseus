"""Score, compact, and summarize SparkStream context packets.

Long autonomous runs generate far more events than should remain in active
context. This script treats conclusions, actions, events, and output chunks as
packets, assigns deterministic importance scores, then produces a compact view:
keep high-value packets, summarize/merge related important packets, and mark
low-value packets as eviction candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_LEDGER = REPORTS / "context_packets.jsonl"
DEFAULT_OUT = REPORTS / "context_packet_ledger.json"
DEFAULT_MAX_PACKETS = 96
DEFAULT_MAX_CHARS = 60000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--ingest-reports", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--max-active-packets", type=int, default=DEFAULT_MAX_PACKETS)
    parser.add_argument("--max-active-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--tail", type=int, default=5000)
    args = parser.parse_args()

    ledger_path = ROOT / args.ledger
    existing = read_jsonl_tail(ledger_path, args.tail)
    existing_ids = {str(item.get("packet_id")) for item in existing if isinstance(item, dict)}
    new_packets: list[dict[str, Any]] = []
    if args.ingest_reports:
        for packet in collect_packets():
            if packet["packet_id"] not in existing_ids:
                new_packets.append(packet)
                existing_ids.add(packet["packet_id"])
        if new_packets:
            append_jsonl_many(ledger_path, new_packets)
            existing.extend(new_packets)

    packets = dedupe_packets([score_packet(packet) for packet in existing if isinstance(packet, dict)])
    compacted = compact_packets(
        packets,
        max_active_packets=max(8, args.max_active_packets),
        max_active_chars=max(2000, args.max_active_chars),
    )
    report = {
        "policy": "sparkstream_context_packets_v0",
        "updated_utc": now(),
        "ledger": str(ledger_path.relative_to(ROOT)).replace("\\", "/"),
        "new_packets": len(new_packets),
        **compacted,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def collect_packets() -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    packets.extend(packetize_candidate_gate(REPORTS / "candidate_promotion_gate.json"))
    packets.extend(packetize_benchmark_ledger(REPORTS / "benchmark_ledger.json"))
    packets.extend(packetize_autonomy_cycle(REPORTS / "autonomy_cycle_last.json"))
    packets.extend(packetize_json_report(REPORTS / "training_ratchet_profile_run.json", "training_profile"))
    packets.extend(packetize_json_report(REPORTS / "autonomy_launch_readiness.json", "launch_readiness"))
    packets.extend(packetize_json_report(REPORTS / "resource_governor.json", "resource_governor"))
    packets.extend(packetize_json_report(REPORTS / "arm_lifecycle_governance.json", "arm_lifecycle"))
    packets.extend(packetize_json_report(REPORTS / "benchmark_seeker_registry.json", "benchmark_seeker"))
    packets.extend(packetize_json_report(REPORTS / "knowledge_source_registry.json", "knowledge_source"))
    packets.extend(packetize_json_report(REPORTS / "synthetic_data_curator.json", "synthetic_data"))
    packets.extend(packetize_json_report(REPORTS / "training_data_sampler.json", "training_data"))
    packets.extend(packetize_json_report(REPORTS / "residual_escrow.json", "residual_escrow"))
    packets.extend(packetize_json_report(REPORTS / "rl_benchmark_registry.json", "rl_registry"))
    packets.extend(packetize_json_report(REPORTS / "legacy_runtime_governance_gate.json", "legacy_runtime_governance"))
    packets.extend(packetize_json_report(REPORTS / "legacy_port_runtime_enforcement.json", "legacy_port_runtime_enforcement"))
    packets.extend(packetize_json_report(REPORTS / "coherence_delirium_gate.json", "coherence_delirium_gate"))
    packets.extend(packetize_json_report(REPORTS / "legacy_training_source_audit.json", "legacy_training_sources"))
    packets.extend(packetize_json_report(REPORTS / "legacy_training_source_sample.json", "legacy_training_sample"))
    packets.extend(packetize_json_report(REPORTS / "legacy_rl_environment_admission.json", "legacy_rl_envs"))
    packets.extend(packetize_json_report(REPORTS / "legacy_rl_smoke_plan.json", "legacy_rl_smoke_plan"))
    packets.extend(packetize_json_report(REPORTS / "trace_fabric_capsule_admission.json", "trace_capsules"))
    packets.extend(packetize_json_report(REPORTS / "trace_fabric_capsule_materialization.json", "trace_capsule_materialization"))
    packets.extend(packetize_json_report(REPORTS / "legacy_adapter_bank_training_plan.json", "legacy_adapter_bank_training_plan"))
    packets.extend(packetize_json_report(REPORTS / "legacy_active_inference_pilot.json", "legacy_active_inference_pilot"))
    packets.extend(packetize_json_report(REPORTS / "whitecell_threat_memory.json", "whitecell_threat_memory"))
    packets.extend(packetize_json_report(REPORTS / "training_data_inventory.json", "data_inventory"))
    packets.extend(packetize_json_report(REPORTS / "personality_context_last.json", "personality_context"))
    packets.extend(packetize_json_report(REPORTS / "personality_drift_eval.json", "personality_drift"))
    packets.extend(packetize_json_report(REPORTS / "belief_update_governance.json", "belief_governance"))
    packets.extend(packetize_json_report(REPORTS / "teacher_oracle_last.json", "teacher"))
    packets.extend(packetize_jsonl_events(REPORTS / "belief_update_ledger.jsonl", "belief_update", limit=30))
    packets.extend(packetize_jsonl_events(REPORTS / "sparkstream_daemon_ledger.jsonl", "daemon_event", limit=40))
    packets.extend(packetize_jsonl_events(REPORTS / "autonomous_goal_ledger.jsonl", "goal_trace", limit=30))
    packets.extend(packetize_jsonl_events(REPORTS / "routing_memory_real_traces.jsonl", "routing_trace", limit=40))
    return packets


def packetize_candidate_gate(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not payload:
        return []
    failed = [
        str(item.get("gate"))
        for item in payload.get("checks", [])
        if isinstance(item, dict) and not item.get("passed")
    ]
    scores = payload.get("scores") or {}
    text = (
        f"Candidate promote={payload.get('promote')} passed={payload.get('passed')}/{payload.get('total')}; "
        f"public={scores.get('public_accuracy')} seed49={scores.get('seed49_regression_accuracy')} "
        f"seed55={scores.get('seed55_frontier_accuracy')}; failed_gates={', '.join(failed) or 'none'}."
    )
    return [
        make_packet(
            source_path=path,
            packet_type="conclusion",
            title="Candidate promotion gate conclusion",
            text=text,
            metadata={
                "promote": payload.get("promote"),
                "failed_gates": failed,
                "scores": scores,
                "evidence_depth": len(payload.get("checks") or []),
                "critical": bool(failed and payload.get("promote") is False),
            },
        )
    ]


def packetize_benchmark_ledger(path: Path) -> list[dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        return []
    packets = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("benchmark_name") or "benchmark"
        lifecycle = row.get("lifecycle")
        score = row.get("score")
        residual = row.get("residual")
        wall = row.get("wall_type")
        threshold = get_path(row, ["graduation_policy", "current_threshold"], None)
        floor = get_path(row, ["graduation_policy", "floor_threshold"], None)
        text = (
            f"{name}: lifecycle={lifecycle}, score={score}, residual={residual}, "
            f"threshold={threshold}, floor={floor}, wall={wall}."
        )
        packets.append(
            make_packet(
                source_path=path,
                packet_type="benchmark",
                title=f"Benchmark {name}",
                text=text,
                metadata={
                    "benchmark": name,
                    "lifecycle": lifecycle,
                    "score": score,
                    "residual": residual,
                    "wall_type": wall,
                    "frontier": lifecycle == "frontier",
                    "critical": lifecycle == "frontier" and isinstance(score, (int, float)) and isinstance(floor, (int, float)) and score < floor,
                },
            )
        )
    return packets


def packetize_autonomy_cycle(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not payload:
        return []
    packets = []
    decision = payload.get("decision") or {}
    commands = [item for item in payload.get("commands", []) if isinstance(item, dict)]
    total_runtime = sum(int(item.get("runtime_ms") or 0) for item in commands)
    failed = [
        item.get("name")
        for item in commands
        if item.get("returncode") not in (0, None) and not item.get("allow_failure", False)
    ]
    allowed_failures = [
        item.get("name")
        for item in commands
        if item.get("returncode") not in (0, None) and item.get("allow_failure", False)
    ]
    packets.append(
        make_packet(
            source_path=path,
            packet_type="conclusion",
            title=f"Autonomy cycle {payload.get('cycle_id')}",
            text=(
                f"Cycle ok={payload.get('ok')} profile={payload.get('profile')} "
                f"decision={decision.get('reason')} teacher_needed={payload.get('teacher_needed')} "
                f"commands={len(commands)} failed={', '.join(str(x) for x in failed) or 'none'} "
                f"allowed_failures={', '.join(str(x) for x in allowed_failures) or 'none'}."
            ),
            metadata={
                "cycle_id": payload.get("cycle_id"),
                "duration_ms": total_runtime,
                "evidence_depth": len(commands),
                "failed_commands": failed,
                "allowed_failure_commands": allowed_failures,
                "critical": bool(failed),
            },
        )
    )
    for command in commands[-30:]:
        name = str(command.get("name") or "command")
        tail = (command.get("stderr_tail") or command.get("stdout_tail") or "").strip()
        text = (
            f"{name}: returncode={command.get('returncode')} runtime_ms={command.get('runtime_ms')} "
            f"skipped={command.get('skipped', False)}. {tail[:800]}"
        )
        packets.append(
            make_packet(
                source_path=path,
                packet_type="action",
                title=f"Command {name}",
                text=text,
                metadata={
                    "duration_ms": int(command.get("runtime_ms") or 0),
                    "returncode": command.get("returncode"),
                    "critical": command.get("returncode") not in (0, None) and not command.get("allow_failure", False),
                },
            )
        )
    return packets


def packetize_json_report(path: Path, packet_type: str) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not payload:
        return []
    title, text, metadata = summarize_payload(path, packet_type, payload)
    return [make_packet(source_path=path, packet_type=packet_type, title=title, text=text, metadata=metadata)]


def packetize_jsonl_events(path: Path, packet_type: str, *, limit: int) -> list[dict[str, Any]]:
    rows = read_jsonl_tail(path, limit)
    packets = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("event") or row.get("goal") or row.get("task") or packet_type)
        text = compact_json(row, 900)
        packets.append(
            make_packet(
                source_path=path,
                packet_type=packet_type,
                title=title,
                text=text,
                metadata={
                    "duration_ms": int(row.get("runtime_ms") or row.get("duration_ms") or 0),
                    "critical": row.get("ok") is False or row.get("returncode") not in (None, 0),
                },
                source_utc=str(row.get("created_utc") or row.get("updated_utc") or ""),
            )
        )
    return packets


def summarize_payload(path: Path, packet_type: str, payload: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if packet_type == "training_profile":
        commands = [item for item in payload.get("commands", []) if isinstance(item, dict)]
        metadata["duration_ms"] = sum(int(item.get("runtime_ms") or 0) for item in commands)
        metadata["evidence_depth"] = len(commands)
        metadata["critical"] = payload.get("ok") is False
        return (
            "Training ratchet profile result",
            f"profile={payload.get('profile')} ok={payload.get('ok')} commands={len(commands)} decision={payload.get('decision')}",
            metadata,
        )
    if packet_type == "launch_readiness":
        blockers = payload.get("blocker_failures") or []
        warnings = payload.get("warning_failures") or []
        metadata["evidence_depth"] = len(blockers) + len(warnings)
        metadata["critical"] = bool(blockers)
        return (
            "Launch readiness",
            (
                f"autonomous_training={payload.get('ready_for_autonomous_training')} "
                f"teacher_run={payload.get('ready_for_teacher_enabled_run')} "
                f"candidate={payload.get('ready_for_candidate_promotion')} "
                f"blockers={len(blockers)} warnings={len(warnings)}"
            ),
            metadata,
        )
    if packet_type == "resource_governor":
        decision = payload.get("decision") or {}
        efficiency = payload.get("efficiency") or {}
        metadata["critical"] = decision.get("can_run_requested_profile") is False
        return (
            "Resource governor",
            (
                f"can_run={decision.get('can_run_requested_profile')} recommended={decision.get('recommended_profile')} "
                f"owner={decision.get('execution_owner')} efficiency={efficiency.get('score')} "
                f"throttle={decision.get('throttle_reasons')}"
            ),
            metadata,
        )
    if packet_type == "arm_lifecycle":
        summary = payload.get("summary") or {}
        proposals = payload.get("proposals") or []
        metadata["evidence_depth"] = len(proposals)
        metadata["critical"] = payload.get("ready_for_long_autonomy") is False
        return (
            "Arm lifecycle governance",
            f"ready={payload.get('ready_for_long_autonomy')} arms={summary.get('arms')} proposals={summary.get('proposal_count')}",
            metadata,
        )
    if packet_type == "benchmark_seeker":
        queued = payload.get("queued_external_candidates") or []
        discovered = payload.get("discovered_external_candidates") or []
        recommendations = payload.get("recommendations") or []
        metadata["evidence_depth"] = len(queued) + len(discovered) + len(recommendations)
        return (
            "Benchmark seeker inventory",
            f"queued={len(queued)} discovered={len(discovered)} recommendations={len(recommendations)}",
            metadata,
        )
    if packet_type == "knowledge_source":
        sources = payload.get("sources") or []
        blocked = [source.get("name") for source in sources if isinstance(source, dict) and not source.get("training_use_allowed")]
        metadata["evidence_depth"] = len(sources)
        return (
            "Knowledge-source gates",
            f"sources={len(sources)} training_blocked={', '.join(str(x) for x in blocked) or 'none'}",
            metadata,
        )
    if packet_type == "synthetic_data":
        verification = payload.get("verification") or {}
        metadata["evidence_depth"] = int(verification.get("total") or 0)
        metadata["critical"] = payload.get("training_ready") is False
        return (
            "Synthetic data curator",
            (
                f"ready={payload.get('training_ready')} accepted={payload.get('accepted_count')} "
                f"blend_ratio={payload.get('blend_synthetic_ratio')} "
                f"quality={verification.get('mean_quality_score')} "
                f"verification_ok={verification.get('ok')}"
            ),
            metadata,
        )
    if packet_type == "residual_escrow":
        summary = payload.get("summary") or {}
        metadata["critical"] = int(summary.get("critical_cluster_count") or 0) > 0
        return (
            "Residual escrow summary",
            compact_json(summary, 900),
            metadata,
        )
    if packet_type == "rl_registry":
        return ("RL benchmark registry", compact_json(payload.get("summary") or {}, 900), metadata)
    if packet_type == "legacy_runtime_governance":
        summary = payload.get("summary") or {}
        metadata["evidence_depth"] = len(payload.get("gates") or [])
        metadata["critical"] = payload.get("trigger_state") == "RED"
        return (
            "Legacy runtime governance gate",
            (
                f"state={payload.get('trigger_state')} teacher={payload.get('ready_for_teacher_work')} "
                f"candidate={payload.get('ready_for_candidate_promotion')} warnings={summary.get('warning_count')} "
                f"failed={compact_json(summary.get('failed_gates') or [], 400)}"
            ),
            metadata,
        )
    if packet_type == "legacy_port_runtime_enforcement":
        summary = payload.get("summary") or {}
        metadata["evidence_depth"] = int(summary.get("effect_records") or 0) + int(summary.get("planforge_nodes") or 0)
        metadata["critical"] = summary.get("trigger_state") == "RED" or not bool(payload.get("ready_for_bounded_autonomy"))
        return (
            "Legacy port runtime enforcement",
            (
                f"state={summary.get('trigger_state')} bounded={payload.get('ready_for_bounded_autonomy')} "
                f"long={payload.get('ready_for_long_autonomy')} self_evolution={payload.get('ready_for_self_evolution')} "
                f"blockers={compact_json(payload.get('blockers') or [], 400)} "
                f"effect_records={summary.get('effect_records')} planforge_nodes={summary.get('planforge_nodes')}"
            ),
            metadata,
        )
    if packet_type == "coherence_delirium_gate":
        metadata["evidence_depth"] = len(payload.get("gates") or [])
        metadata["critical"] = not bool(payload.get("allows_long_autonomy"))
        return (
            "Coherence/delirium gate",
            (
                f"state={payload.get('trigger_state')} source={payload.get('source_trigger_state')} "
                f"coherence={payload.get('coherence_score')} delirium={payload.get('delirium_score')} "
                f"long_autonomy={payload.get('allows_long_autonomy')} "
                f"candidate={payload.get('allows_candidate_promotion')} "
                f"self_edit={payload.get('allows_self_edit')}"
            ),
            metadata,
        )
    if packet_type == "legacy_training_sources":
        summary = payload.get("summary") or {}
        metadata["evidence_depth"] = int(summary.get("ready_local_verified") or 0)
        metadata["critical"] = payload.get("trigger_state") == "RED"
        return (
            "Legacy training-source admission",
            (
                f"state={payload.get('trigger_state')} ready={summary.get('ready_local_verified')} "
                f"serious={summary.get('serious_training_ready')} hash_mismatches={summary.get('hash_mismatches')} "
                f"redacted={summary.get('reference_answers_redacted')}/{summary.get('reference_answers_seen')}"
            ),
            metadata,
        )
    if packet_type == "legacy_training_sample":
        summary = payload.get("summary") or {}
        metadata["evidence_depth"] = int(summary.get("sample_rows") or 0)
        metadata["critical"] = payload.get("trigger_state") == "RED"
        return (
            "Legacy training tiny sample",
            (
                f"state={payload.get('trigger_state')} rows={summary.get('sample_rows')} "
                f"sources={summary.get('selected_sources')} lanes={compact_json(summary.get('lane_counts') or {}, 400)}"
            ),
            metadata,
        )
    if packet_type == "legacy_rl_envs":
        summary = payload.get("summary") or {}
        metadata["evidence_depth"] = int(summary.get("environments") or 0)
        metadata["critical"] = payload.get("trigger_state") == "RED"
        return (
            "Legacy RL environment admission",
            (
                f"state={payload.get('trigger_state')} envs={summary.get('environments')} "
                f"p0={summary.get('p0_smoke_lane')} drone={summary.get('drone_envs')} "
                f"hardware_gated={summary.get('hardware_gated_envs')}"
            ),
            metadata,
        )
    if packet_type == "legacy_rl_smoke_plan":
        summary = payload.get("summary") or {}
        metadata["evidence_depth"] = int(summary.get("planned_envs") or 0)
        metadata["critical"] = payload.get("trigger_state") == "RED"
        return (
            "Legacy RL smoke plan",
            (
                f"state={payload.get('trigger_state')} planned={summary.get('planned_envs')} "
                f"ready={summary.get('ready_for_seeded_smoke')} pending={summary.get('pending_dependency')} "
                f"source_present_pending_install={summary.get('source_present_pending_install')} "
                f"runner_pending_adapter={summary.get('runner_pending_adapter')} "
                f"hardware_gated={summary.get('hardware_gated_not_executable')}"
            ),
            metadata,
        )
    if packet_type == "trace_capsules":
        summary = payload.get("summary") or {}
        metadata["evidence_depth"] = int(summary.get("accepted_metadata_only") or 0)
        metadata["critical"] = payload.get("trigger_state") == "RED"
        return (
            "Trace-fabric capsule admission",
            (
                f"state={payload.get('trigger_state')} capsules={summary.get('capsules')} "
                f"accepted={summary.get('accepted_metadata_only')} quarantined={summary.get('quarantined')} "
                f"missing_sources={summary.get('missing_sources')}"
            ),
            metadata,
        )
    if packet_type == "trace_capsule_materialization":
        summary = payload.get("summary") or {}
        metadata["evidence_depth"] = int(summary.get("materialized_rows") or 0)
        metadata["critical"] = payload.get("trigger_state") == "RED"
        return (
            "Trace-fabric capsule materialization",
            (
                f"state={payload.get('trigger_state')} rows={summary.get('materialized_rows')} "
                f"lanes={compact_json(summary.get('lane_counts') or {}, 400)} "
                f"raw={summary.get('raw_payload_rows')} rejections={compact_json(summary.get('rejections') or {}, 400)}"
            ),
            metadata,
        )
    if packet_type == "legacy_adapter_bank_training_plan":
        summary = payload.get("summary") or {}
        metadata["evidence_depth"] = int(summary.get("plan_rows") or 0)
        metadata["critical"] = payload.get("trigger_state") == "RED"
        return (
            "Legacy adapter-bank training plan",
            (
                f"state={payload.get('trigger_state')} zero_param_ready={payload.get('ready_for_zero_param_dry_run')} "
                f"activation_ready={payload.get('ready_for_adapter_activation')} rows={summary.get('plan_rows')} "
                f"selected={compact_json(summary.get('selected_adapters') or [], 400)} "
                f"zero_param={compact_json(summary.get('zero_param_lanes') or [], 400)}"
            ),
            metadata,
        )
    if packet_type == "legacy_active_inference_pilot":
        summary = payload.get("summary") or {}
        metadata["evidence_depth"] = int(summary.get("steps") or 0) + int(summary.get("accepted_belief_updates") or 0)
        metadata["critical"] = payload.get("trigger_state") == "RED"
        return (
            "Legacy active-inference pilot",
            (
                f"state={payload.get('trigger_state')} ready={payload.get('ready_for_world_model_training_signal')} "
                f"error={summary.get('mean_prediction_error')} rankings={summary.get('action_rankings')} "
                f"belief_updates={summary.get('accepted_belief_updates')} replay={payload.get('replay_id')}"
            ),
            metadata,
        )
    if packet_type == "whitecell_threat_memory":
        patterns = [row for row in payload.get("threat_patterns", []) if isinstance(row, dict)]
        active = [str(row.get("pattern_id")) for row in patterns if row.get("active")]
        blockers = [
            str(row.get("pattern_id"))
            for row in patterns
            if row.get("active") and row.get("action") == "block_and_escalate"
        ]
        metadata["evidence_depth"] = len(patterns)
        metadata["critical"] = bool(blockers)
        return (
            "WhiteCell local threat memory",
            (
                f"state={payload.get('trigger_state')} local_only={payload.get('local_only')} "
                f"active={compact_json(active, 400)} block_and_escalate={compact_json(blockers, 400)}"
            ),
            metadata,
        )
    if packet_type == "data_inventory":
        return ("Training data inventory", compact_json(payload.get("summary") or {}, 900), metadata)
    if packet_type == "personality_context":
        summary = payload.get("summary") or {}
        metadata["evidence_depth"] = int(summary.get("selected_cards") or 0)
        metadata["critical"] = payload.get("status") != "ready"
        return (
            "Personality context",
            (
                f"status={payload.get('status')} selected_cards={summary.get('selected_cards')} "
                f"hard_invariants={summary.get('hard_safety_invariants')} anti_drift={summary.get('anti_drift_rules')}"
            ),
            metadata,
        )
    if packet_type == "personality_drift":
        summary = payload.get("summary") or {}
        metadata["evidence_depth"] = int(summary.get("total") or 0)
        metadata["critical"] = not bool(payload.get("passed"))
        return (
            "Personality drift eval",
            f"passed={payload.get('passed')} total={summary.get('total')} score={summary.get('average_score')}",
            metadata,
        )
    if packet_type == "belief_governance":
        summary = payload.get("summary") or {}
        metadata["critical"] = int(summary.get("quarantined") or 0) > 0
        return (
            "Belief update governance",
            f"status={payload.get('status')} ledger_entries={summary.get('ledger_entries')} quarantined={summary.get('quarantined')}",
            metadata,
        )
    if packet_type == "teacher":
        response = payload.get("response_json") or payload.get("response_text") or payload
        metadata["critical"] = payload.get("status") not in (None, "queued", "ok", "completed")
        metadata["evidence_depth"] = 3
        return ("Teacher oracle result", compact_json(response, 1200), metadata)
    return (path.stem, compact_json(payload, 1200), metadata)


def make_packet(
    *,
    source_path: Path,
    packet_type: str,
    title: str,
    text: str,
    metadata: dict[str, Any],
    source_utc: str = "",
) -> dict[str, Any]:
    source_ref = str(source_path.relative_to(ROOT)).replace("\\", "/")
    source_stat = safe_stat(source_path)
    created = source_utc or modified_utc(source_stat)
    packet_basis = f"{source_ref}\n{source_stat.get('mtime_ns')}\n{packet_type}\n{title}\n{text[:1000]}"
    return {
        "packet_id": stable_id(packet_basis),
        "created_utc": created,
        "ingested_utc": now(),
        "source_path": source_ref,
        "packet_type": packet_type,
        "title": title,
        "text": " ".join(str(text).split()),
        "metadata": metadata,
        "estimated_chars": len(str(text)),
        "estimated_tokens": max(1, len(str(text)) // 4),
    }


def score_packet(packet: dict[str, Any]) -> dict[str, Any]:
    metadata = packet.get("metadata") if isinstance(packet.get("metadata"), dict) else {}
    packet_type = str(packet.get("packet_type") or "")
    text = f"{packet.get('title', '')} {packet.get('text', '')}".lower()
    duration_ms = float(metadata.get("duration_ms") or 0.0)
    evidence_depth = float(metadata.get("evidence_depth") or 0.0)
    kind_weight = {
        "conclusion": 3.2,
        "benchmark": 2.5,
        "launch_readiness": 2.4,
        "residual_escrow": 2.3,
        "teacher": 2.2,
        "resource_governor": 1.9,
        "arm_lifecycle": 1.9,
        "training_profile": 1.8,
        "knowledge_source": 1.7,
        "synthetic_data": 2.0,
        "benchmark_seeker": 1.5,
        "action": 1.2,
        "routing_trace": 1.1,
        "goal_trace": 1.1,
        "daemon_event": 0.7,
        "data_inventory": 0.6,
        "rl_registry": 0.9,
        "legacy_runtime_governance": 2.4,
        "legacy_port_runtime_enforcement": 2.8,
        "coherence_delirium_gate": 2.6,
        "legacy_training_sources": 2.2,
        "legacy_training_sample": 2.1,
        "legacy_rl_envs": 1.8,
        "legacy_rl_smoke_plan": 1.8,
        "trace_capsules": 2.0,
        "trace_capsule_materialization": 2.1,
        "legacy_adapter_bank_training_plan": 2.2,
        "legacy_active_inference_pilot": 2.2,
        "whitecell_threat_memory": 2.3,
    }.get(packet_type, 1.0)
    duration_score = min(1.8, math.log1p(duration_ms / 1000.0) / 2.0)
    evidence_score = min(1.2, math.log1p(evidence_depth) / 2.0)
    keyword_score = 0.0
    for keyword, bonus in [
        ("frontier", 0.7),
        ("failed", 0.6),
        ("blocked", 0.6),
        ("critical", 0.8),
        ("residual", 0.5),
        ("promote=false", 0.4),
        ("teacher", 0.4),
        ("cuda", 0.25),
        ("grokipedia", 0.25),
        ("synthetic", 0.35),
        ("quality", 0.2),
    ]:
        if keyword in text:
            keyword_score += bonus
    critical_score = 1.0 if metadata.get("critical") else 0.0
    frontier_score = 0.8 if metadata.get("frontier") else 0.0
    size_penalty = min(0.8, float(packet.get("estimated_tokens") or 0) / 4000.0)
    recency_score = recency_bonus(str(packet.get("created_utc") or ""))
    score = kind_weight + duration_score + evidence_score + keyword_score + critical_score + frontier_score + recency_score - size_penalty
    packet = dict(packet)
    packet["importance"] = {
        "score": round(score, 4),
        "kind_weight": kind_weight,
        "duration_score": round(duration_score, 4),
        "evidence_score": round(evidence_score, 4),
        "keyword_score": round(keyword_score, 4),
        "critical_score": critical_score,
        "frontier_score": frontier_score,
        "recency_score": round(recency_score, 4),
        "size_penalty": round(size_penalty, 4),
    }
    return packet


def compact_packets(
    packets: list[dict[str, Any]],
    *,
    max_active_packets: int,
    max_active_chars: int,
) -> dict[str, Any]:
    ranked = sorted(
        packets,
        key=lambda item: (float(get_path(item, ["importance", "score"], 0.0)), str(item.get("created_utc") or "")),
        reverse=True,
    )
    active: list[dict[str, Any]] = []
    active_chars = 0
    for packet in ranked:
        packet_chars = int(packet.get("estimated_chars") or len(str(packet.get("text") or "")))
        if len(active) >= max_active_packets or active_chars + packet_chars > max_active_chars:
            continue
        active.append(packet)
        active_chars += packet_chars
    active_ids = {packet["packet_id"] for packet in active}
    dropped = [packet for packet in ranked if packet.get("packet_id") not in active_ids]
    summaries = summarize_groups(active)
    context_view = sorted(
        summaries + active[: max_active_packets // 2],
        key=lambda item: float(get_path(item, ["importance", "score"], 0.0)),
        reverse=True,
    )
    return {
        "summary": {
            "packet_count": len(packets),
            "active_packet_count": len(active),
            "summary_packet_count": len(summaries),
            "drop_candidate_count": len(dropped),
            "active_chars": active_chars,
            "max_active_packets": max_active_packets,
            "max_active_chars": max_active_chars,
            "top_score": get_path(ranked[0], ["importance", "score"], None) if ranked else None,
        },
        "active_packets": sanitize_packets(active, 80),
        "summary_packets": summaries,
        "drop_candidates": sanitize_packets(dropped[-80:], 80),
        "context_view": sanitize_packets(context_view, 80),
    }


def dedupe_packets(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the newest packet for the same logical source/title slot."""

    by_slot: dict[tuple[str, str, str], dict[str, Any]] = {}
    for packet in packets:
        slot = (
            str(packet.get("source_path") or ""),
            str(packet.get("packet_type") or ""),
            str(packet.get("title") or ""),
        )
        current = by_slot.get(slot)
        if current is None or str(packet.get("ingested_utc") or packet.get("created_utc") or "") >= str(
            current.get("ingested_utc") or current.get("created_utc") or ""
        ):
            by_slot[slot] = packet
    return list(by_slot.values())


def summarize_groups(active: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for packet in active:
        key = str(packet.get("packet_type") or "packet")
        groups.setdefault(key, []).append(packet)
    summaries = []
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        top_rows = sorted(rows, key=lambda item: float(get_path(item, ["importance", "score"], 0.0)), reverse=True)[:6]
        bullets = [f"- {row.get('title')}: {str(row.get('text') or '')[:260]}" for row in top_rows]
        score = max(float(get_path(row, ["importance", "score"], 0.0)) for row in top_rows) + min(0.6, len(rows) / 20)
        text = f"Merged {len(rows)} {key} packets:\n" + "\n".join(bullets)
        summaries.append(
            {
                "packet_id": stable_id(f"summary\n{key}\n" + "\n".join(str(row.get("packet_id")) for row in top_rows)),
                "created_utc": now(),
                "source_path": "reports/context_packets.jsonl",
                "packet_type": "summary",
                "title": f"Context summary: {key}",
                "text": text,
                "merged_packet_ids": [row.get("packet_id") for row in top_rows],
                "estimated_chars": len(text),
                "estimated_tokens": max(1, len(text) // 4),
                "importance": {
                    "score": round(score, 4),
                    "kind_weight": 2.5,
                    "duration_score": 0.0,
                    "evidence_score": min(1.2, math.log1p(len(rows)) / 2.0),
                    "keyword_score": 0.0,
                    "critical_score": 0.0,
                    "frontier_score": 0.0,
                    "recency_score": 0.0,
                    "size_penalty": 0.0,
                },
            }
        )
    return sorted(summaries, key=lambda item: float(get_path(item, ["importance", "score"], 0.0)), reverse=True)


def sanitize_packets(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    sanitized = []
    for row in rows[:limit]:
        item = dict(row)
        if len(str(item.get("text") or "")) > 1800:
            item["text"] = str(item["text"])[:1800] + "..."
        sanitized.append(item)
    return sanitized


def recency_bonus(created_utc: str) -> float:
    try:
        created = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age_hours = max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 3600.0)
    return max(0.0, 0.5 - min(0.5, age_hours / 48.0))


def safe_stat(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"mtime_ns": 0, "mtime": 0.0}
    return {"mtime_ns": stat.st_mtime_ns, "mtime": stat.st_mtime}


def modified_utc(stat: dict[str, Any]) -> str:
    return datetime.fromtimestamp(float(stat.get("mtime") or time.time()), timezone.utc).isoformat()


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


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def append_jsonl_many(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def compact_json(value: Any, limit: int) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + "..."


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
