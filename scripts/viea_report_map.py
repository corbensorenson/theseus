"""Emit the VIEA subsystem map for Theseus reports and dashboards.

VIEA is the top-level architecture contract. This report makes the mapping
machine-readable so dashboards and docs can say which subsystem each report
serves, and so learning proof stays attached to broad public transfer rather
than scaffold health.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

SUBSYSTEMS: list[dict[str, Any]] = [
    {
        "id": "viea_autonomy_spine",
        "purpose": "Run goal/command -> executor -> artifact kernel -> runtime packet -> verification -> feedback ratchet -> next action as the autonomy control path.",
        "reports": [
            "reports/viea_autonomy_spine.json",
            "reports/feedback_action_queue.json",
            "reports/viea_action_executor.json",
            "reports/broad_transfer_action_queue.json",
            "reports/vacation_mode_supervisor.json",
            "reports/vacation_mode_failure_triage.json",
            "reports/vacation_mode_repair_action_queue.json",
        ],
        "bundle_files": [],
        "dashboard_surfaces": ["VIEA OS panel", "feedback action queue", "broad transfer closure queue"],
        "promotion_evidence": False,
        "rule": "The spine can execute local control actions and queue training pressure; it cannot convert scaffold health into student promotion.",
    },
    {
        "id": "structured_command_layer",
        "purpose": "Convert raw user goals into explicit command contracts.",
        "reports": ["reports/reality_manipulator.json", "reports/viea_command_executor.json", "reports/viea_autonomy_spine.json"],
        "bundle_files": ["reports/reality_manipulator/latest_world/command_contract.json"],
        "dashboard_surfaces": ["Reality Manipulator card", "goal/command launch controls"],
        "promotion_evidence": False,
    },
    {
        "id": "artifact_kernel",
        "purpose": "Persist VIEA worlds, commands, artifacts, claims, critiques, tools, benchmarks, residuals, releases, and feedback in SQLite.",
        "reports": ["reports/viea_artifact_kernel.json"],
        "bundle_files": ["reports/viea_artifact_kernel.sqlite"],
        "dashboard_surfaces": ["artifact kernel", "world object database", "VIEA object counts"],
        "promotion_evidence": False,
    },
    {
        "id": "artifact_graph",
        "purpose": "Preserve worlds, artifacts, provenance, release manifests, and feedback.",
        "reports": ["reports/reality_manipulator.json", "reports/genesis_kernel/report.json", "reports/viea_artifact_kernel.json"],
        "bundle_files": [
            "reports/reality_manipulator/latest_world/world.json",
            "reports/reality_manipulator/latest_world/artifacts.json",
            "reports/reality_manipulator/latest_world/release_manifest.json",
            "reports/reality_manipulator/latest_world/resource_log.jsonl",
        ],
        "dashboard_surfaces": ["Reality Manipulator card", "Genesis/artifact status"],
        "promotion_evidence": False,
    },
    {
        "id": "claim_and_verification_ledger",
        "purpose": "Separate verified, supported, speculative, blocked, and critique-backed claims.",
        "reports": [
            "reports/reality_manipulator.json",
            "reports/deterministic_taming_stack.json",
            "reports/student_first_evidence_audit.json",
        ],
        "bundle_files": [
            "reports/reality_manipulator/latest_world/claim_ledger.json",
            "reports/reality_manipulator/latest_world/critique_log.json",
        ],
        "dashboard_surfaces": ["learning truth gates", "claim/critique status"],
        "promotion_evidence": False,
    },
    {
        "id": "orchestrator_router",
        "purpose": "Route work through bounded specialist modules with permissions and memory scopes.",
        "reports": [
            "reports/viea_command_executor.json",
            "reports/octopus_router_report.json",
            "reports/routing_memory.json",
            "reports/arm_lifecycle_governance.json",
            "reports/feedback_action_queue.json",
        ],
        "bundle_files": ["reports/reality_manipulator/latest_world/specialist_lifecycle.json"],
        "dashboard_surfaces": ["Octopus routing", "arm lifecycle", "selected arms"],
        "promotion_evidence": False,
    },
    {
        "id": "specialist_modules",
        "purpose": "Provide bounded writing, code, safety, benchmark, memory, CAD, chip, and fabrication capability.",
        "reports": [
            "reports/arm_lifecycle_governance.json",
            "reports/cell_lifecycle.json",
            "reports/arm_sucker_registry.json",
        ],
        "bundle_files": ["reports/reality_manipulator/latest_world/specialist_lifecycle.json"],
        "dashboard_surfaces": ["arm health", "cell lifecycle", "sucker registry"],
        "promotion_evidence": False,
    },
    {
        "id": "workflow_to_tool_compiler",
        "purpose": "Compile repeated successful workflows into verified reusable tools without distilling benchmark answers.",
        "reports": [
            "reports/workflow_tool_compiler_v2.json",
            "reports/loop_closure_harvester.json",
            "reports/loop_closure_tool_promoter.json",
            "reports/tool_registry.json",
            "reports/workflow_routing_traces.jsonl",
        ],
        "bundle_files": ["reports/reality_manipulator/latest_world/workflow_tool_metrics.json"],
        "dashboard_surfaces": ["tool registry", "loop closure", "workflow traces"],
        "promotion_evidence": False,
    },
    {
        "id": "evaluation_ratchet",
        "purpose": "Move benchmark frontiers, preserve regressions, escrow residuals, and avoid stale score semantics.",
        "reports": [
            "reports/broad_transfer_closure.json",
            "reports/broad_transfer_action_queue.json",
            "reports/learning_scoreboard.json",
            "reports/benchmaxx_curriculum.json",
            "reports/broad_transfer_matrix.json",
            "reports/multi_turn_conversation_benchmark.json",
            "reports/personality_runtime_audit.json",
            "reports/transfer_generalization_audit.json",
            "reports/broad_code_calibration_scheduler.json",
            "reports/candidate_promotion_gate.json",
            "reports/residual_escrow.json",
        ],
        "bundle_files": [],
        "dashboard_surfaces": ["learning scoreboard", "Benchmaxx", "candidate gate", "residual escrow"],
        "promotion_evidence": False,
    },
    {
        "id": "transfer_generalization_guard",
        "purpose": "Prevent narrow benchmark-specific benchmaxxing by measuring cross-card spread, shared residual concepts, STS per-card causality, and source-name curriculum concentration.",
        "reports": [
            "reports/transfer_generalization_audit.json",
            "reports/broad_transfer_matrix.json",
            "reports/code_residual_curriculum.json",
            "reports/broad_code_calibration_scheduler.json",
        ],
        "bundle_files": [],
        "dashboard_surfaces": ["Benchmaxx generalization", "shared residual concepts", "overfit risks"],
        "promotion_evidence": False,
        "rule": "Private pressure should target source-agnostic concepts that transfer to receiver cards; one benchmark card above floor is not broad capability.",
    },
    {
        "id": "runtime_adapters",
        "purpose": "Target outputs to digital, hardware/chip, fabrication, robotic, organizational, or spatial runtimes.",
        "reports": [
            "reports/digital_runtime_adapter.json",
            "reports/reality_manipulator.json",
            "reports/benchmark_adapter_factory.json",
            "reports/resource_pantry.json",
            "reports/hive_scheduler.json",
        ],
        "bundle_files": ["reports/reality_manipulator/latest_world/world.json"],
        "dashboard_surfaces": ["adapter factory", "resource pantry", "Hive runtime paths"],
        "promotion_evidence": False,
    },
    {
        "id": "feedback_loop",
        "purpose": "Feed runtime, benchmark, residual, user, and deployment outcomes back into artifacts and future pressure.",
        "reports": [
            "reports/feedback_ratchet.json",
            "reports/feedback_action_queue.json",
            "reports/viea_action_executor.json",
            "reports/learning_scoreboard.json",
            "reports/autonomy_ledger.jsonl",
            "reports/sparkstream_daemon_ledger.jsonl",
            "reports/viea_action_execution_ledger.jsonl",
            "reports/checkpoint_registry.json",
        ],
        "bundle_files": ["reports/reality_manipulator/latest_world/feedback_plan.md"],
        "dashboard_surfaces": ["autonomy ledger", "checkpoint registry", "daemon status"],
        "promotion_evidence": False,
    },
    {
        "id": "student_learning_proof_layer",
        "purpose": "Prove actual student learning with clean token-level generation and broad public transfer.",
        "reports": [
            "reports/broad_transfer_matrix.json",
            "reports/real_code_benchmark_graduation.json",
            "reports/student_first_evidence_audit.json",
            "reports/code_lm_closure.json",
        ],
        "bundle_files": [],
        "dashboard_surfaces": ["broad transfer matrix", "student-first audit", "real code graduation"],
        "promotion_evidence": True,
        "rule": "VIEA scaffold health cannot promote the student; promotion requires clean token-level student generation and broad public transfer.",
    },
    {
        "id": "private_repo_repair_curriculum",
        "purpose": "Train long-horizon programming through private hidden-test repo-repair traces without public benchmark leakage.",
        "reports": [
            "reports/private_repo_repair_curriculum.json",
            "reports/repo_repair_main_curriculum.json",
            "reports/viea_repo_repair_learner.json",
            "reports/repo_repair_trace_checkpoint.json",
            "reports/long_horizon_programming_curriculum.json",
        ],
        "bundle_files": [],
        "dashboard_surfaces": ["private SWE-style curriculum", "repo repair traces", "private hidden-test pressure"],
        "promotion_evidence": False,
        "rule": "Private repo-repair pressure can train and diagnose; public SWE-style surfaces remain calibration-only.",
    },
    {
        "id": "symliquid_substrate",
        "purpose": "Keep SymLiquid as the compact recurrent/state substrate for routing memory, residual clustering, decoding, tool selection, STS conditioning, control, and autonomy state.",
        "reports": ["reports/symliquid_substrate_map.json", "reports/symliquid_state_engine_queue.json", "reports/symliquid_state_engine.json"],
        "bundle_files": [],
        "dashboard_surfaces": ["SymLiquid substrate map", "router memory", "STS/state decoder health"],
        "promotion_evidence": False,
    },
    {
        "id": "teacher_as_architect",
        "purpose": "Let the teacher read residual clusters and propose architecture experiment specs only, never benchmark answers.",
        "reports": ["reports/teacher_architect_loop.json", "reports/teacher_architect_closure.json", "reports/teacher_architect_experiment_runner.json", "reports/architecture_guidance_loop.json"],
        "bundle_files": [],
        "dashboard_surfaces": ["teacher architecture queue", "architecture experiment specs"],
        "promotion_evidence": False,
        "rule": "Teacher output is guidance only; promote or rollback happens through private eval and public calibration.",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/viea_report_map.json")
    parser.add_argument("--markdown-out", default="reports/viea_report_map.md")
    args = parser.parse_args()

    rows = [enrich(row) for row in SUBSYSTEMS]
    payload = {
        "policy": "project_theseus_viea_report_map_v1",
        "created_utc": now(),
        "canonical_doc": "docs/VIEA.md",
        "trigger_state": trigger_state(rows),
        "subsystems": rows,
        "summary": {
            "subsystem_count": len(rows),
            "promotion_evidence_subsystems": [row["id"] for row in rows if row.get("promotion_evidence")],
            "missing_required_report_count": sum(1 for row in rows for item in row["report_status"] if item["required"] and not item["exists"]),
            "missing_bundle_file_count": sum(1 for row in rows for item in row["bundle_status"] if item["required"] and not item["exists"]),
            "dashboard_endpoint": "http://127.0.0.1:8787",
            "hive_endpoint": "http://127.0.0.1:8791",
        },
        "rules": {
            "scaffold_vs_learning": "VIEA reports can prove governance, routing, and artifact preservation; only student-first broad transfer proves learned public-code capability.",
            "public_benchmarks": "public benchmark data remains calibration-only and must not enter private training as solutions or hidden tests",
            "teacher": "teacher proposes architecture experiments only; no public answers, hidden tests, or apply mode",
        },
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, payload)
    write_text(ROOT / args.markdown_out, render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] in {"GREEN", "YELLOW"} else 2


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["report_status"] = [path_status(path, required=True) for path in row.get("reports", [])]
    out["bundle_status"] = [path_status(path, required=True) for path in row.get("bundle_files", [])]
    out["coverage_state"] = "GREEN" if all(item["exists"] for item in out["report_status"] + out["bundle_status"]) else "YELLOW"
    return out


def path_status(path: str, *, required: bool) -> dict[str, Any]:
    full = ROOT / path
    return {
        "path": path,
        "exists": full.exists(),
        "required": required,
        "kind": "jsonl" if path.endswith(".jsonl") else "markdown" if path.endswith(".md") else "json" if path.endswith(".json") else "file",
        "size_bytes": full.stat().st_size if full.exists() and full.is_file() else 0,
    }


def trigger_state(rows: list[dict[str, Any]]) -> str:
    missing_learning = [
        item
        for row in rows
        if row.get("promotion_evidence")
        for item in row["report_status"]
        if item["required"] and not item["exists"]
    ]
    if missing_learning:
        return "RED"
    missing = [
        item
        for row in rows
        for item in row["report_status"] + row["bundle_status"]
        if item["required"] and not item["exists"]
    ]
    return "YELLOW" if missing else "GREEN"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# VIEA Report Map",
        "",
        f"- trigger_state: `{payload['trigger_state']}`",
        f"- canonical_doc: `{payload['canonical_doc']}`",
        f"- dashboard: `{payload['summary']['dashboard_endpoint']}`",
        f"- hive: `{payload['summary']['hive_endpoint']}`",
        "",
        "## Core Rule",
        "",
        payload["rules"]["scaffold_vs_learning"],
        "",
        "## Subsystems",
        "",
    ]
    for row in payload["subsystems"]:
        lines.append(f"### {row['id']}")
        lines.append("")
        lines.append(row["purpose"])
        lines.append("")
        lines.append(f"- coverage_state: `{row['coverage_state']}`")
        lines.append(f"- promotion_evidence: `{row.get('promotion_evidence', False)}`")
        if row.get("rule"):
            lines.append(f"- rule: {row['rule']}")
        lines.append("- reports:")
        for item in row["report_status"]:
            mark = "present" if item["exists"] else "missing"
            lines.append(f"  - `{item['path']}`: {mark}")
        if row["bundle_status"]:
            lines.append("- bundle files:")
            for item in row["bundle_status"]:
                mark = "present" if item["exists"] else "missing"
                lines.append(f"  - `{item['path']}`: {mark}")
        lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
