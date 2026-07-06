"""CLI entrypoint for the source-agnostic high-transfer curriculum scheduler."""

from __future__ import annotations

import argparse
import json
import time

import report_evidence_store
from high_transfer_scheduler_builder import build_concepts
from high_transfer_scheduler_common import (
    CONFIGS,
    DEFAULT_MARKDOWN,
    DEFAULT_OUT,
    DEFAULT_TASKS,
    REPORTS,
    ROOT,
    now,
    read_json,
    resolve,
    write_json,
    write_jsonl,
    write_text,
)
from high_transfer_scheduler_rotation import (
    best_conversation_report,
    best_or_file,
    latest_architecture_guidance,
    latest_report,
    render_markdown,
    task_from_concept,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--tasks-out", default=str(DEFAULT_TASKS.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    report_evidence_store.ingest_default_reports()
    transfer = best_or_file("transfer_generalization", REPORTS / "transfer_generalization_audit.json")
    broad = best_or_file("broad_transfer", REPORTS / "broad_transfer_matrix.json")
    guidance = latest_architecture_guidance()
    conversation = best_conversation_report(
        REPORTS / "high_transfer_multi_turn_conversation.json",
        REPORTS / "multi_turn_conversation_benchmark.json",
    )
    conversation_hard = latest_report(REPORTS / "high_transfer_multi_turn_conversation_hard.json")
    conversation_hard_v2 = latest_report(REPORTS / "high_transfer_multi_turn_conversation_hard_v2.json")
    conversation_hard_v3 = latest_report(REPORTS / "high_transfer_multi_turn_conversation_hard_v3.json")
    conversation_hard_v4 = latest_report(REPORTS / "high_transfer_multi_turn_conversation_hard_v4.json")
    conversation_pantry = read_json(REPORTS / "open_conversation_training_pantry.json", {})
    repo = read_json(REPORTS / "viea_repo_repair_learner.json", {})
    board_game = read_json(REPORTS / "board_game_rl_benchmark.json", {})
    pufferlib4_rl = read_json(REPORTS / "pufferlib4_rl_lane.json", {})
    long_horizon = read_json(REPORTS / "high_transfer_long_horizon_tool_use.json", {})
    cross_domain_capsules = read_json(REPORTS / "cross_domain_sts_capsules.json", {})
    type_contract = read_json(REPORTS / "type_contract_diagnostic.json", {})
    autonomy_policy = read_json(CONFIGS / "autonomy_policy.json", {})
    concepts = build_concepts(
        transfer,
        broad,
        guidance,
        conversation,
        conversation_hard,
        conversation_hard_v2,
        conversation_hard_v3,
        conversation_hard_v4,
        conversation_pantry,
        repo,
        board_game,
        pufferlib4_rl,
        long_horizon,
        cross_domain_capsules,
        type_contract,
        autonomy_policy,
    )
    tasks = [task_from_concept(row) for row in concepts if row.get("status") == "ready"]
    report = {
        "policy": "project_theseus_high_transfer_curriculum_scheduler_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if tasks else "YELLOW",
        "summary": {
            "concept_count": len(concepts),
            "ready_task_count": len(tasks),
            "critical_task_count": sum(1 for row in tasks if row.get("priority") == "critical"),
            "donor_receiver_checks": sum(len(row.get("transfer_checks") or []) for row in concepts),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "concepts": concepts,
        "tasks": tasks,
        "rules": {
            "benchmarks": "public benchmark cards are receiver calibration surfaces only",
            "curriculum": "private source-agnostic concept pressure is preferred over benchmark-name pressure",
            "generalist_rotation": "after a flat code receiver calibration, rotate hard-v3/hard-v2 conversation, board-game RL, PufferLib/Ocean RL, long-horizon tool use, repo repair, and cross-domain STS capsules before more code churn",
            "conversation_graduation": "conversation lanes must pass large/hard calibrations before regression-only; saturated hard_v2 graduates into hard_v3 rather than rerunning the same surface",
            "promotion": "only donor/receiver improvement with no leakage can support capability claims",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.tasks_out), tasks)
    report_evidence_store.ingest_report_path(report_evidence_store.DEFAULT_DB, resolve(args.out), payload=report)
    print(json.dumps(report, indent=2))
    return 2 if report["trigger_state"] == "RED" else 0





if __name__ == "__main__":
    raise SystemExit(main())
