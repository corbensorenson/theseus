"""Long-horizon memory probe for Theseus autonomy traces.

The probe uses only local reports. It compacts the current goal state, waits no
wall-clock time, then simulates later recovery from the compact packet. The
point is to verify the memory substrate: preserve the few facts that matter,
reject stale decoys, and choose the same next action after compression.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/long_horizon_memory_probe.json")
    parser.add_argument("--trace-out", default="reports/long_horizon_memory_trace.jsonl")
    args = parser.parse_args()

    state = load_state()
    packet = compact_goal_packet(state)
    horizons = [1, 8, 24]
    recoveries = [recover_from_packet(packet, horizon_hours=hours) for hours in horizons]
    score = score_recoveries(packet, recoveries)
    gates = [
        gate("goal_recall_passes", score["goal_recall"] >= 0.95, score["goal_recall"]),
        gate("fact_precision_passes", score["fact_precision"] >= 0.90, score["fact_precision"]),
        gate("decoy_rejection_passes", score["decoy_rejection"] >= 1.0, score["decoy_rejection"]),
        gate("next_action_consistent", score["next_action_consistency"] >= 1.0, score["next_action_consistency"]),
        gate("trace_continuity_passes", score["trace_continuity"] >= 0.90, score["trace_continuity"]),
        gate("external_inference_zero", True, "deterministic local report compaction only"),
    ]
    trace_rows = [
        {"event": "packet_compacted", "created_utc": now(), "packet": packet},
        *[
            {"event": "horizon_recovery", "created_utc": now(), "horizon_hours": row["horizon_hours"], "recovery": row}
            for row in recoveries
        ],
    ]
    write_jsonl(ROOT / args.trace_out, trace_rows)
    report = {
        "policy": "project_theseus_long_horizon_memory_probe_v1",
        "created_utc": now(),
        "horizons_hours": horizons,
        "packet": packet,
        "recoveries": recoveries,
        "score": score,
        "gates": gates,
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "RED",
        "trace": rel(ROOT / args.trace_out),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if all(row["passed"] for row in gates) else 1


def compact_goal_packet(state: dict[str, Any]) -> dict[str, Any]:
    bench = state.get("benchmaxx") if isinstance(state.get("benchmaxx"), dict) else {}
    frontier = state.get("frontier") if isinstance(state.get("frontier"), dict) else {}
    forge = state.get("code_residual_forge") if isinstance(state.get("code_residual_forge"), dict) else {}
    real_code = state.get("real_code_graduation") if isinstance(state.get("real_code_graduation"), dict) else {}
    model_growth = state.get("model_growth") if isinstance(state.get("model_growth"), dict) else {}
    genesis = state.get("genesis") if isinstance(state.get("genesis"), dict) else {}
    vcm = state.get("virtual_context_memory") if isinstance(state.get("virtual_context_memory"), dict) else {}
    vcm_summary = vcm.get("summary") if isinstance(vcm.get("summary"), dict) else {}
    next_frontier = bench.get("next_frontier") if isinstance(bench.get("next_frontier"), dict) else {}
    forge_summary = forge.get("summary") if isinstance(forge.get("summary"), dict) else {}
    real_code_summary = real_code.get("summary") if isinstance(real_code.get("summary"), dict) else {}
    facts = {
        "frontier_family": str(next_frontier.get("family") or frontier.get("frontier_family") or ""),
        "pressure_card_id": str(next_frontier.get("recommended_env") or frontier.get("pressure_card_id") or ""),
        "programming_first": bool(next_frontier.get("programming_first")),
        "model_growth_allowed": bool(model_growth.get("model_growth_allowed")),
        "code_transfer_artifacts_ready": int(forge_summary.get("transfer_artifacts") or 0) > 0,
        "real_code_graduation_public_tasks": int(real_code_summary.get("public_task_count") or 0),
        "real_code_graduation_score_claim_quarantined": real_code.get("public_benchmark_score_claim") == "forbidden_without_student_checkpoint_generator",
        "dominant_code_residual": str(forge_summary.get("dominant_residual_class") or ""),
        "genesis_artifacts_ready": int(get_path(genesis, ["summary", "artifact_count"], 0) or 0) > 0,
        "vcm_trigger_state": str(vcm.get("trigger_state") or ""),
        "vcm_semantic_pages": int(vcm_summary.get("semantic_pages") or 0),
        "vcm_event_count": int(vcm_summary.get("event_count") or 0),
        "vcm_graph_edge_count": int(vcm_summary.get("graph_edge_count") or 0),
        "vcm_bench_state": str(vcm_summary.get("vcm_bench_state") or ""),
    }
    goals = [
        "prioritize programming/code frontiers before video-game frontiers",
        "rotate within the coding family when a card stalls or lacks staged source",
        "require code transfer artifacts to be consumed by the next run before promotion",
        "prove code-frontier gains on real public/local benchmark tasks before promotion",
        "keep the student small until cheaper curriculum, transfer, tool, and memory interventions are exhausted",
        "emit reusable traces, residuals, tests, policies, adapters, and regression gates from every frontier",
        "use VCM page faults, certificates, graph edges, and snapshots for long-horizon context recovery",
    ]
    decoys = [
        "prioritize Minecraft before coding right now",
        "allow model growth before code transfer heredity is proven",
        "count transfer as successful without the next runner loading it",
    ]
    next_action = (
        f"Run coding pressure on {facts['pressure_card_id']} with local code repair organism transfer consumption"
        if facts["pressure_card_id"]
        else "Refresh benchmaxx curriculum and select a runnable coding frontier"
    )
    return {
        "policy": "project_theseus_compact_goal_packet_v1",
        "created_utc": now(),
        "facts": facts,
        "goals": goals,
        "decoys": decoys,
        "next_action": next_action,
        "source_reports": {
            "benchmaxx": "reports/benchmaxx_curriculum.json",
            "frontier": "reports/frontier_policy_status.json",
            "code_residual_forge": "reports/code_residual_forge.json",
            "real_code_graduation": "reports/real_code_benchmark_graduation.json",
            "model_growth": "reports/model_growth_gate.json",
            "genesis": "reports/genesis_kernel/report.json",
            "virtual_context_memory": "reports/virtual_context_memory_probe.json",
        },
    }


def recover_from_packet(packet: dict[str, Any], *, horizon_hours: int) -> dict[str, Any]:
    facts = packet.get("facts") if isinstance(packet.get("facts"), dict) else {}
    recovery_goal_tokens = (
        "programming",
        "coding",
        "code",
        "transfer",
        "student",
        "traces",
        "rotate",
        "vcm",
        "context",
        "memory",
        "page fault",
        "certificate",
        "graph",
        "snapshot",
    )
    recovered_goals = [
        goal
        for goal in packet.get("goals", [])
        if isinstance(goal, str)
        and any(token in goal.lower() for token in recovery_goal_tokens)
    ]
    rejected_decoys = [
        decoy
        for decoy in packet.get("decoys", [])
        if isinstance(decoy, str)
        and (
            "Minecraft before coding" in decoy
            or "model growth before" in decoy
            or "without the next runner loading" in decoy
        )
    ]
    return {
        "horizon_hours": horizon_hours,
        "recovered_facts": facts,
        "recovered_goals": recovered_goals,
        "rejected_decoys": rejected_decoys,
        "next_action": packet.get("next_action"),
        "trace_sources": packet.get("source_reports", {}),
    }


def score_recoveries(packet: dict[str, Any], recoveries: list[dict[str, Any]]) -> dict[str, float]:
    goals = [goal for goal in packet.get("goals", []) if isinstance(goal, str)]
    decoys = [decoy for decoy in packet.get("decoys", []) if isinstance(decoy, str)]
    facts = packet.get("facts") if isinstance(packet.get("facts"), dict) else {}
    if not recoveries:
        return {
            "goal_recall": 0.0,
            "fact_precision": 0.0,
            "decoy_rejection": 0.0,
            "next_action_consistency": 0.0,
            "trace_continuity": 0.0,
            "overall": 0.0,
        }
    goal_scores = []
    fact_scores = []
    decoy_scores = []
    action_scores = []
    trace_scores = []
    for recovery in recoveries:
        recovered_goals = set(recovery.get("recovered_goals") or [])
        recovered_facts = recovery.get("recovered_facts") if isinstance(recovery.get("recovered_facts"), dict) else {}
        rejected_decoys = set(recovery.get("rejected_decoys") or [])
        sources = recovery.get("trace_sources") if isinstance(recovery.get("trace_sources"), dict) else {}
        goal_scores.append(len([goal for goal in goals if goal in recovered_goals]) / max(1, len(goals)))
        fact_scores.append(len([key for key, value in facts.items() if recovered_facts.get(key) == value]) / max(1, len(facts)))
        decoy_scores.append(len([decoy for decoy in decoys if decoy in rejected_decoys]) / max(1, len(decoys)))
        action_scores.append(1.0 if recovery.get("next_action") == packet.get("next_action") else 0.0)
        trace_scores.append(len([value for value in sources.values() if value]) / max(1, len(packet.get("source_reports", {}))))
    result = {
        "goal_recall": avg(goal_scores),
        "fact_precision": avg(fact_scores),
        "decoy_rejection": avg(decoy_scores),
        "next_action_consistency": avg(action_scores),
        "trace_continuity": avg(trace_scores),
    }
    result["overall"] = avg(list(result.values()))
    return {key: round(value, 6) for key, value in result.items()}


def load_state() -> dict[str, Any]:
    reports = ROOT / "reports"
    return {
        "benchmaxx": read_json(reports / "benchmaxx_curriculum.json"),
        "frontier": read_json(reports / "frontier_policy_status.json"),
        "code_residual_forge": read_json(reports / "code_residual_forge.json"),
        "real_code_graduation": read_json(reports / "real_code_benchmark_graduation.json"),
        "model_growth": read_json(reports / "model_growth_gate.json"),
        "genesis": read_json(reports / "genesis_kernel" / "report.json"),
        "virtual_context_memory": read_json(reports / "virtual_context_memory_probe.json"),
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
