#!/usr/bin/env python3
"""Build cross-domain STS skill capsules from non-code learning lanes.

The goal is to let conversation, games, long-horizon tool use, and repo repair
produce reusable state/skill pressure without copying raw traces into training.
Capsules are metadata-first: they capture the transferable lesson, residual
focus, target arms, and a short STS conditioning sentence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import theseus_runtime


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "cross_domain_sts_capsules.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "cross_domain_sts_capsules.md"
DEFAULT_CAPSULES_OUT = ROOT / "data" / "training_sources" / "cross_domain_sts_capsules.jsonl"
DEFAULT_DATA_DIR = Path(theseus_runtime.runtime_report(create=False)["paths"]["data_dir"]["path"])
DEFAULT_STS_OUT = DEFAULT_DATA_DIR / "cross_domain_sts" / "cross_domain_sts_streams.jsonl"

DEFAULT_SOURCE_REPORTS = [
    REPORTS / "high_transfer_multi_turn_conversation_hard_v4.json",
    REPORTS / "high_transfer_multi_turn_conversation_hard.json",
    REPORTS / "high_transfer_multi_turn_conversation_hard_v2.json",
    REPORTS / "high_transfer_multi_turn_conversation_hard_v3.json",
    REPORTS / "high_transfer_multi_turn_conversation.json",
    REPORTS / "board_game_rl_benchmark.json",
    REPORTS / "board_game_learned_policy.json",
    REPORTS / "pufferlib4_capability_probe.json",
    REPORTS / "pufferlib4_rl_lane.json",
    REPORTS / "high_transfer_long_horizon_tool_use.json",
    REPORTS / "hive_work_board_executor.json",
    REPORTS / "high_transfer_repo_repair_learner.json",
]

SKILL_TARGETS = {
    "goal_tracking": ["conversation_arm", "tool_use_arm", "repo_repair_arm", "code_generation_arm"],
    "state_memory": ["conversation_arm", "board_game_rl_arm", "pufferlib4_rl_arm", "tool_use_arm", "code_generation_arm"],
    "legal_action_masking": ["board_game_rl_arm", "pufferlib4_rl_arm", "tool_use_arm", "code_generation_arm"],
    "edge_conditions": ["code_generation_arm", "repo_repair_arm", "conversation_arm", "board_game_rl_arm", "pufferlib4_rl_arm"],
    "branch_planning": ["code_generation_arm", "repo_repair_arm", "board_game_rl_arm", "pufferlib4_rl_arm"],
    "repair_after_failure": ["repo_repair_arm", "tool_use_arm", "code_generation_arm"],
    "interface_contracts": ["code_generation_arm", "repo_repair_arm", "tool_use_arm"],
    "multi_step_execution": ["tool_use_arm", "repo_repair_arm", "conversation_arm", "board_game_rl_arm"],
    "personality_core_adherence": ["conversation_arm", "operator_copilot_arm", "tool_use_arm"],
    "evidence_grounding": ["conversation_arm", "benchmark_ratchet_arm", "repo_repair_arm"],
    "reward_credit_assignment": ["board_game_rl_arm", "pufferlib4_rl_arm", "code_generation_arm", "tool_use_arm"],
    "reset_step_contracts": ["pufferlib4_rl_arm", "tool_use_arm", "code_generation_arm"],
    "rollout_replay": ["pufferlib4_rl_arm", "board_game_rl_arm", "repo_repair_arm", "code_generation_arm"],
    "policy_value_trace": ["pufferlib4_rl_arm", "board_game_rl_arm", "symliquid_state_engine"],
}

CONSUMER_CONTRACTS = {
    "code_generation_arm": {
        "consumer": "code_lm_closure",
        "control_surface": "decoder_control_hints_or_sts_streams",
        "ablation": "same_seed_sts_on_vs_off_or_contract_guided_vs_baseline",
        "evidence_report": "reports/decoder_v2_private_ablation_gate.json",
    },
    "repo_repair_arm": {
        "consumer": "viea_repo_repair_learner",
        "control_surface": "repo_repair_trace_checkpoint_and_code_lm_rows",
        "ablation": "repo_repair_rows_on_vs_off_private_hidden_tests",
        "evidence_report": "reports/high_transfer_repo_repair_learner.json",
    },
    "tool_use_arm": {
        "consumer": "long_horizon_tool_use_benchmark",
        "control_surface": "tool_selection_retry_resume_policy",
        "ablation": "capsule_conditioned_tool_policy_vs_baseline",
        "evidence_report": "reports/high_transfer_long_horizon_tool_use.json",
    },
    "conversation_arm": {
        "consumer": "multi_turn_conversation_benchmark",
        "control_surface": "conversation_memory_and_correction_policy",
        "ablation": "capsule_conditioned_conversation_vs_baseline",
        "evidence_report": "reports/high_transfer_multi_turn_conversation_hard_v4.json",
    },
    "board_game_rl_arm": {
        "consumer": "board_game_learned_policy",
        "control_surface": "legal_action_mask_policy_value_rows",
        "ablation": "learned_policy_vs_random_and_simple_baselines",
        "evidence_report": "reports/board_game_learned_policy.json",
    },
    "pufferlib4_rl_arm": {
        "consumer": "pufferlib4_rl_lane",
        "control_surface": "reset_step_policy_value_trace",
        "ablation": "pufferlib_policy_rollout_vs_baseline",
        "evidence_report": "reports/pufferlib4_rl_lane.json",
    },
    "symliquid_state_engine": {
        "consumer": "symliquid_state_engine",
        "control_surface": "route_weights_and_decoder_control_hints",
        "ablation": "route_weight_change_plus_downstream_delta",
        "evidence_report": "reports/symliquid_state_engine.json",
    },
    "octopus_router": {
        "consumer": "octopus_router",
        "control_surface": "arm_selection_prior",
        "ablation": "router_choice_distribution_change_plus_task_delta",
        "evidence_report": "reports/octopus_router.json",
    },
    "benchmark_ratchet_arm": {
        "consumer": "benchmark_ratchet",
        "control_surface": "frontier_rotation_and_regression_marking",
        "ablation": "frontier_rotation_with_capsules_vs_without",
        "evidence_report": "reports/high_transfer_curriculum_scheduler.json",
    },
    "operator_copilot_arm": {
        "consumer": "personality_runtime_audit",
        "control_surface": "operator_handoff_and_status_policy",
        "ablation": "personality_context_on_vs_off",
        "evidence_report": "reports/personality_runtime_audit.json",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT.relative_to(ROOT)))
    parser.add_argument("--capsules-out", default=str(DEFAULT_CAPSULES_OUT.relative_to(ROOT)))
    parser.add_argument("--sts-out", default=str(DEFAULT_STS_OUT))
    parser.add_argument("--max-capsules", type=int, default=256)
    parser.add_argument("--source-report", action="append", default=[])
    args = parser.parse_args()

    source_reports = [resolve(item) for item in args.source_report] if args.source_report else DEFAULT_SOURCE_REPORTS
    capsules: list[dict[str, Any]] = []
    source_status: list[dict[str, Any]] = []
    for path in source_reports:
        report = read_json(path, {})
        source_status.append(
            {
                "path": rel_or_abs(path),
                "exists": path.exists(),
                "policy": report.get("policy"),
                "trigger_state": report.get("trigger_state") or report.get("status"),
                "sha256": sha256_file(path) if path.exists() else "",
            }
        )
        if not report:
            continue
        capsules.extend(capsules_from_report(path, report))

    deduped = dedupe_capsules(capsules)[: max(1, args.max_capsules)]
    sts_rows = [sts_row(capsule) for capsule in deduped]
    capsules_out = resolve(args.capsules_out)
    sts_out = resolve(args.sts_out)
    write_jsonl(capsules_out, deduped)
    write_jsonl(sts_out, sts_rows)

    lane_counts = Counter(str(row.get("lane") or "unknown") for row in deduped)
    skill_counts = Counter(str(row.get("skill") or "unknown") for row in deduped)
    target_counts = Counter(target for row in deduped for target in row.get("transfer_targets", []))
    causal = causal_evidence()
    unsafe = [row for row in deduped if row.get("public_benchmark_training_data_used") or row.get("external_inference_calls")]
    missing_consumers = [
        row.get("capsule_id")
        for row in deduped
        if not row.get("causal_consumers") or any(not consumer_declared(item) for item in row.get("causal_consumers", []))
    ]
    gates = [
        gate("source_reports_present", any(row["exists"] for row in source_status), source_status),
        gate("capsules_present", len(deduped) > 0, len(deduped)),
        gate("sts_rows_written", len(sts_rows) == len(deduped) and len(sts_rows) > 0, rel_or_abs(sts_out)),
        gate("causal_consumers_named", not missing_consumers, missing_consumers[:20]),
        gate("ablation_contracts_present", all(row.get("causal_ablation_contracts") for row in deduped), "every capsule names how it must prove transfer"),
        gate("measured_transfer_effect_present", causal["measured_transfer_effect"], causal),
        gate("no_public_benchmark_training_data", not unsafe, [row.get("capsule_id") for row in unsafe[:20]]),
        gate("metadata_only_capsules", all(not row.get("raw_payload_copied") for row in deduped), "raw payloads are not copied"),
        gate("external_inference_zero", all(int(row.get("external_inference_calls") or 0) == 0 for row in deduped), 0),
    ]
    report = {
        "policy": "project_theseus_cross_domain_sts_capsules_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "YELLOW",
        "summary": {
            "source_report_count": len(source_status),
            "source_reports_present": sum(1 for row in source_status if row["exists"]),
            "capsule_count": len(deduped),
            "sts_row_count": len(sts_rows),
            "lane_counts": dict(lane_counts),
            "skill_counts": dict(skill_counts),
            "transfer_target_counts": dict(target_counts),
            "causal_transfer": causal,
            "capsules_out": rel_or_abs(capsules_out),
            "sts_out": rel_or_abs(sts_out),
            "external_inference_calls": 0,
        },
        "source_reports": source_status,
        "capsules": deduped[:80],
        "gates": gates,
        "rules": {
            "public_benchmarks": "public benchmark score reports can condition metadata only; public solutions/tests never become training rows",
            "raw_payloads": "raw traces are referenced by hash/path and summarized into skill capsules",
            "transfer": "capsules route state/skill lessons into STS and arm conditioning, not score claims",
            "causality": "a capsule is not mature until a named consumer changes behavior and an A/B or baseline comparison shows downstream lift",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def capsules_from_report(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    policy = str(report.get("policy") or "")
    if policy == "project_theseus_multi_turn_conversation_benchmark_v1":
        return conversation_capsules(path, report)
    if policy == "project_theseus_board_game_rl_benchmark_v1":
        return board_game_capsules(path, report)
    if policy == "project_theseus_board_game_learned_policy_v1":
        return board_game_learned_policy_capsules(path, report)
    if policy == "project_theseus_pufferlib4_capability_probe_v1":
        return pufferlib4_probe_capsules(path, report)
    if policy == "project_theseus_pufferlib4_rl_lane_v1":
        return pufferlib4_lane_capsules(path, report)
    if policy == "project_theseus_long_horizon_tool_use_benchmark_v1":
        return long_horizon_tool_use_capsules(path, report)
    if policy == "project_theseus_hive_work_board_executor_v1":
        return long_horizon_capsules(path, report)
    if policy == "project_theseus_viea_repo_repair_learner_v1":
        return repo_repair_capsules(path, report)
    return []


def conversation_capsules(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    suite_mode = str(summary.get("suite_mode") or "unknown")
    accuracy = number(summary.get("accuracy"))
    turn_count = int(number(summary.get("turn_count")))
    personality_ready = int(number(summary.get("personality_context_ready_turns")))
    rows = [
        capsule(
            path,
            lane="conversation",
            skill="state_memory",
            summary=f"{suite_mode} conversation carried state across {turn_count} turns at accuracy {accuracy:.3f}.",
            residual="conversation_frontier_regression" if bool(summary.get("graduated")) else "conversation_frontier_active",
            evidence={"accuracy": accuracy, "case_count": summary.get("case_count"), "turn_count": turn_count},
        ),
        capsule(
            path,
            lane="conversation",
            skill="personality_core_adherence",
            summary=f"Personality context was ready for {personality_ready}/{turn_count} turns.",
            residual="personality_context_gap" if personality_ready != turn_count else "personality_context_preserved",
            evidence={"personality_ready_turns": personality_ready, "turn_count": turn_count},
        ),
        capsule(
            path,
            lane="conversation",
            skill="evidence_grounding",
            summary="Conversation benchmark pressures status answers to cite reports, gates, uncertainty, and residuals.",
            residual="status_overclaim_guard",
            evidence={"suite_mode": suite_mode, "passed": report.get("passed")},
        ),
        capsule(
            path,
            lane="conversation",
            skill="goal_tracking",
            summary="Multi-turn sessions pressure corrections, target updates, and active objective carryover.",
            residual="correction_memory_and_goal_tracking",
            evidence={"case_sources": summary.get("case_sources")},
        ),
    ]
    failures = report.get("failures") if isinstance(report.get("failures"), list) else []
    for idx, failure in enumerate(failures[:16]):
        rows.append(
            capsule(
                path,
                lane="conversation",
                skill="repair_after_failure",
                summary=f"Conversation residual needs repair: {str(failure)[:180]}",
                residual=str(failure)[:160],
                evidence={"failure_index": idx},
                quality=0.78,
            )
        )
    return rows


def board_game_capsules(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    games = report.get("games") if isinstance(report.get("games"), dict) else {}
    trace_path = resolve(str(get_path(report, ["outputs", "traces"], "reports/board_game_rl_traces.jsonl")))
    trace_rows = read_jsonl(trace_path)
    for game, payload in sorted(games.items()):
        if not isinstance(payload, dict) or payload.get("skipped"):
            continue
        game_traces = [row for row in trace_rows if str(row.get("game") or "") == str(game)]
        diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
        rating_delta = payload.get("rating_delta") if isinstance(payload.get("rating_delta"), dict) else {}
        rows.append(
            capsule(
                path,
                lane=f"{game}_rl",
                skill="legal_action_masking",
                summary=f"{game} legal-action and tactical diagnostics stayed gated before Elo evidence was accepted.",
                residual="illegal_action_or_rule_gap" if any(not bool(d.get("ok")) for d in diagnostics.values() if isinstance(d, dict)) else "legal_action_gate_green",
                evidence={"diagnostics": compact(diagnostics), "rating_delta": rating_delta},
            )
        )
        rows.append(
            capsule(
                path,
                lane=f"{game}_rl",
                skill="branch_planning",
                summary=f"{game} self-play compares policy branches under legal moves and terminal reward.",
                residual="policy_branch_selection",
                evidence={"matches": len(payload.get("matches") or []), "ratings": payload.get("ratings")},
            )
        )
        rows.append(
            capsule(
                path,
                lane=f"{game}_rl",
                skill="reward_credit_assignment",
                summary=f"{game} Elo deltas provide trend evidence for local policy improvement, not single-run promotion.",
                residual="sparse_reward_credit_assignment",
                evidence={"rating_delta": rating_delta},
            )
        )
        rows.append(
            capsule(
                path,
                lane=f"{game}_rl",
                skill="multi_step_execution",
                summary=f"{game} self-play emitted {len(game_traces)} replay traces with legal actions, terminal state, and per-ply policy choices.",
                residual="self_play_trace_to_replayable_policy_memory",
                evidence=board_game_trace_summary(game_traces),
                quality=0.84,
            )
        )
        rows.append(
            capsule(
                path,
                lane=f"{game}_rl",
                skill="state_memory",
                summary=f"{game} policy traces pressure compact state memory across long action sequences rather than one-step scoring.",
                residual="long_state_rollout_memory",
                evidence={
                    "trace_report": rel_or_abs(trace_path),
                    "trace_sha256": sha256_file(trace_path) if trace_path.exists() else "",
                    "sampled_trace_count": min(len(game_traces), 16),
                },
                quality=0.83,
            )
        )
        rows.extend(board_game_failure_capsules(path, game, game_traces))
    return rows


def long_horizon_tool_use_capsules(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    cases = report.get("cases") if isinstance(report.get("cases"), list) else []
    rows = [
        capsule(
            path,
            lane="long_horizon_tool_use",
            skill="multi_step_execution",
            summary=f"Local terminal/tool-use lane completed {summary.get('passed_cases')}/{summary.get('case_count')} private cases at pass rate {summary.get('pass_rate')}.",
            residual="long_horizon_tool_use_frontier",
            evidence={
                "case_count": summary.get("case_count"),
                "pass_rate": summary.get("pass_rate"),
                "trace_rows": summary.get("trace_rows"),
                "sts_rows": summary.get("sts_rows"),
            },
            quality=0.86,
        ),
        capsule(
            path,
            lane="long_horizon_tool_use",
            skill="state_memory",
            summary="Checkpoint/resume cases pressure persistent task state rather than one-shot answer generation.",
            residual="checkpoint_resume_state",
            evidence={"outputs": report.get("outputs")},
            quality=0.84,
        ),
        capsule(
            path,
            lane="long_horizon_tool_use",
            skill="repair_after_failure",
            summary="Retry/recovery cases pressure bounded repair after a failed local action.",
            residual="bounded_retry_recovery",
            evidence={"residuals": summary.get("residuals")},
            quality=0.84,
        ),
        capsule(
            path,
            lane="long_horizon_tool_use",
            skill="interface_contracts",
            summary="Terminal pipeline and evidence-routing cases pressure explicit input/output contracts and artifact routing.",
            residual="tool_interface_and_evidence_contract",
            evidence={"skills": summary.get("skills")},
            quality=0.82,
        ),
    ]
    for idx, case in enumerate(cases[:24]):
        if not isinstance(case, dict):
            continue
        rows.append(
            capsule(
                path,
                lane="long_horizon_tool_use",
                skill=str(case.get("skill") or "multi_step_execution"),
                summary=f"Tool-use case {case.get('case_id')} residual {case.get('residual')}.",
                residual=str(case.get("residual") or "unknown_tool_use_residual"),
                evidence={"case_id": case.get("case_id"), "passed": case.get("passed"), "evidence": case.get("evidence")},
                quality=0.78 if case.get("passed") else 0.72,
            )
        )
    return rows


def board_game_trace_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"trace_count": 0}
    plies = [int(number(row.get("plies"))) for row in rows]
    winners = Counter(str(row.get("winner") or "unknown") for row in rows)
    terminations = Counter(str(row.get("termination") or "unknown") for row in rows)
    policies = Counter()
    for row in rows[:32]:
        policies[str(row.get("white_policy") or "")] += 1
        policies[str(row.get("black_policy") or "")] += 1
    return {
        "trace_count": len(rows),
        "avg_plies": round(sum(plies) / max(1, len(plies)), 2),
        "max_plies": max(plies) if plies else 0,
        "winner_counts": dict(winners),
        "termination_counts": dict(terminations),
        "policy_counts_sample": dict(+policies),
    }


def board_game_failure_capsules(path: Path, game: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[:12]):
        winner = str(row.get("winner") or "")
        if winner not in {"draw", "unknown", ""}:
            continue
        out.append(
            capsule(
                path,
                lane=f"{game}_rl",
                skill="repair_after_failure",
                summary=f"{game} self-play trace ended without a decisive policy win; keep the replay as tactical residual metadata.",
                residual=f"{game}_draw_or_unclear_terminal_policy_trace",
                evidence={
                    "trace_index": idx,
                    "plies": row.get("plies"),
                    "termination": row.get("termination"),
                    "white_policy": row.get("white_policy"),
                    "black_policy": row.get("black_policy"),
                    "trace_digest": stable_hash({k: row.get(k) for k in ['game', 'seed', 'winner', 'plies', 'final_fen', 'white_policy', 'black_policy']})[:16],
                },
                quality=0.79,
            )
        )
    return out[:6]


def board_game_learned_policy_capsules(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    weights = summary.get("skill_weights") if isinstance(summary.get("skill_weights"), dict) else {}
    policy_train_rows = summary.get("policy_train_row_count")
    for card in report.get("policy_cards", []):
        if not isinstance(card, dict):
            continue
        game = str(card.get("game") or "board_game")
        rows.append(
            capsule(
                path,
                lane=f"{game}_learned_policy",
                skill="legal_action_masking",
                summary=f"{game} learned-policy card captured {card.get('legal_action_mask_examples')} legal action examples from self-play.",
                residual="legal_action_mask_to_program_interface_transfer",
                evidence={
                    "best_policy": card.get("best_policy_by_current_elo"),
                    "trace_count": card.get("trace_count"),
                    "skill_weight": weights.get("legal_action_masking"),
                    "policy_train_row_count": policy_train_rows,
                },
                quality=0.87,
            )
        )
        rows.append(
            capsule(
                path,
                lane=f"{game}_learned_policy",
                skill="branch_planning",
                summary=str(card.get("learned_control_lesson") or "")[:260],
                residual="state_action_branch_planning_from_self_play",
                evidence={
                    "policy_counts": card.get("policy_counts"),
                    "terminal_counts": card.get("terminal_counts"),
                    "skill_weight": weights.get("branch_planning"),
                },
                quality=0.86,
            )
        )
        residuals = card.get("tactical_residuals") if isinstance(card.get("tactical_residuals"), dict) else {}
        if residuals and "none_observed" not in residuals:
            rows.append(
                capsule(
                    path,
                    lane=f"{game}_learned_policy",
                    skill="repair_after_failure",
                    summary=f"{game} learned-policy card preserved tactical residuals for replay and repair.",
                    residual=";".join(sorted(str(key) for key in residuals))[:220],
                    evidence={"tactical_residuals": residuals, "skill_weight": weights.get("repair_after_failure")},
                    quality=0.82,
                )
            )
    for hint in report.get("skill_capsule_hints", []):
        if not isinstance(hint, dict):
            continue
        skill = str(hint.get("skill") or "")
        if not skill or skill not in SKILL_TARGETS:
            continue
        rows.append(
            capsule(
                path,
                lane="board_game_learned_policy",
                skill=skill,
                summary=f"Board-game learned policy assigned transfer weight {hint.get('weight')} to {skill}.",
                residual="learned_policy_weight_to_sts_conditioning",
                evidence={"weight": hint.get("weight"), "transfer_targets": hint.get("transfer_targets")},
                quality=0.81,
            )
        )
    return rows


def pufferlib4_probe_capsules(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    blocker_ids = [str(row.get("id") or "") for row in blockers if isinstance(row, dict)]
    return [
        capsule(
            path,
            lane="pufferlib4_rl",
            skill="interface_contracts",
            summary=f"PufferLib 4 probe found puffer import={summary.get('pufferlib_import_ok')}, native backend={summary.get('native_backend_ok')}, Ocean envs={summary.get('ocean_env_count')}.",
            residual=";".join(blocker_ids[:8]) or "pufferlib4_runtime_admitted",
            evidence={
                "native_backend_ok": summary.get("native_backend_ok"),
                "ocean_env_count": summary.get("ocean_env_count"),
                "atari_enabled": summary.get("atari_enabled"),
                "podman_connected": summary.get("podman_connected"),
            },
            quality=0.82,
        ),
        capsule(
            path,
            lane="pufferlib4_rl",
            skill="reset_step_contracts",
            summary="Puffer/Ocean admission keeps reset/step/native-backend runtime facts explicit before training is allowed.",
            residual="runtime_gate_before_rollout",
            evidence={"trigger_state": report.get("trigger_state"), "blockers": blocker_ids[:8]},
            quality=0.82,
        ),
    ]


def pufferlib4_lane_capsules(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    targets = report.get("transfer_targets") if isinstance(report.get("transfer_targets"), list) else []
    policy_learning = bool(
        summary.get("policy_learning_evidence")
        or summary.get("native_policy_learning_evidence")
        or summary.get("fallback_policy_learning_evidence")
    )
    skill_map = {
        "legal_action_masks": "legal_action_masking",
        "state_memory": "state_memory",
        "sparse_reward_credit_assignment": "reward_credit_assignment",
        "branching_plan_selection": "branch_planning",
        "reset_step_contracts": "reset_step_contracts",
        "rollout_replay": "rollout_replay",
        "policy_value_trace": "policy_value_trace",
        "repair_after_loss": "repair_after_failure",
    }
    rows: list[dict[str, Any]] = []
    for target in targets:
        skill = skill_map.get(str(target), str(target))
        rows.append(
            capsule(
                path,
                lane="pufferlib4_rl",
                skill=skill,
                summary=f"PufferLib 4 RL lane preserved transfer pressure for {target}; native backend ready={summary.get('native_backend_ready')}.",
                residual=str(summary.get("improvement_signal") or "pufferlib4_rl_transfer_pressure"),
                evidence={
                    "native_backend_ready": summary.get("native_backend_ready"),
                    "policy_learning_evidence": policy_learning,
                    "policy_learning_backend": summary.get("policy_learning_backend"),
                    "policy_accuracy_delta": summary.get("policy_accuracy_delta"),
                    "policy_rollout_reward_delta": summary.get("policy_rollout_reward_delta"),
                    "atari_enabled": summary.get("atari_enabled"),
                    "capsule_count": summary.get("capsule_count"),
                    "outputs": report.get("outputs"),
                },
                quality=0.86 if policy_learning else 0.78,
            )
        )
    return rows


def long_horizon_capsules(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    selected = report.get("selected") if isinstance(report.get("selected"), list) else []
    results = report.get("results") if isinstance(report.get("results"), list) else []
    rows = [
        capsule(
            path,
            lane="long_horizon_tool_use",
            skill="multi_step_execution",
            summary="Hive board execution records selected tasks, durable status, retries, evidence, and next assignment.",
            residual="long_horizon_resume_and_status",
            evidence={
                "ready_tasks": summary.get("ready_tasks"),
                "selected_tasks": summary.get("selected_tasks"),
                "executed_tasks": summary.get("executed_tasks"),
            },
        ),
        capsule(
            path,
            lane="long_horizon_tool_use",
            skill="goal_tracking",
            summary="Board-selected work preserves active objectives across daemon cycles and report refreshes.",
            residual="objective_continuity",
            evidence={"selected_titles": [str(row.get("title"))[:80] for row in selected[:5] if isinstance(row, dict)]},
        ),
    ]
    for row in results[:16]:
        if not isinstance(row, dict):
            continue
        skill = "repair_after_failure" if row.get("status") in {"failed", "blocked"} else "interface_contracts"
        rows.append(
            capsule(
                path,
                lane="long_horizon_tool_use",
                skill=skill,
                summary=f"Board task {row.get('title')} finished with status {row.get('status')} and reason {row.get('reason')}.",
                residual=str(row.get("reason") or row.get("status") or "board_task_result"),
                evidence={"task_id": row.get("task_id"), "runtime_ms": row.get("runtime_ms")},
                quality=0.80,
            )
        )
    return rows


def repo_repair_capsules(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return [
        capsule(
            path,
            lane="repo_repair",
            skill="repair_after_failure",
            summary=f"Repo repair produced {summary.get('validated_private_trace_count')} validated private traces and {summary.get('code_lm_row_count')} Code LM rows.",
            residual="private_repo_patch_test_repair",
            evidence={"category_counts": summary.get("category_counts")},
        ),
        capsule(
            path,
            lane="repo_repair",
            skill="interface_contracts",
            summary="Repo repair ties bug inference, patch shape, private tests, residual labels, and trace checkpointing into one contract.",
            residual="patch_interface_and_test_contract",
            evidence={"trace_out": summary.get("trace_out"), "checkpoint_out": summary.get("checkpoint_out")},
        ),
        capsule(
            path,
            lane="repo_repair",
            skill="multi_step_execution",
            summary="Repo repair is the bridge from single-function generation to multi-step programming autonomy.",
            residual="repo_snapshot_patch_test_trace_loop",
            evidence={"task_count": summary.get("task_count"), "trace_count": summary.get("trace_count")},
        ),
    ]


def capsule(
    source_path: Path,
    *,
    lane: str,
    skill: str,
    summary: str,
    residual: str,
    evidence: dict[str, Any],
    quality: float = 0.86,
    utility: float = 0.88,
) -> dict[str, Any]:
    source_sha = sha256_file(source_path) if source_path.exists() else ""
    body = {
        "source": rel_or_abs(source_path),
        "lane": lane,
        "skill": skill,
        "summary": summary,
        "residual": residual,
        "evidence": evidence,
    }
    capsule_id = "xsts_" + stable_hash(body)[:16]
    targets = SKILL_TARGETS.get(skill, ["symliquid_state_engine", "octopus_router"])
    consumers = consumer_contracts_for(targets)
    return {
        "capsule_id": capsule_id,
        "policy": "project_theseus_cross_domain_sts_capsule_v1",
        "created_utc": now(),
        "source_report": rel_or_abs(source_path),
        "source_sha256": source_sha,
        "lane": lane,
        "skill": skill,
        "transfer_targets": targets,
        "summary": summary[:500],
        "residual_focus": residual[:240],
        "quality_score": round(float(quality), 3),
        "utility_score": round(float(utility), 3),
        "training_use_state": "sts_conditioning_candidate_metadata_only",
        "contamination_boundary": "private_local_non_code_metadata; public benchmark reports are calibration-only and solution/test content is not copied",
        "causal_consumers": consumers,
        "causal_ablation_contracts": [row["ablation"] for row in consumers if row.get("ablation")],
        "causal_status": "pending_ablation",
        "raw_payload_copied": False,
        "public_benchmark_training_data_used": False,
        "external_inference_calls": 0,
        "evidence_digest": stable_hash(evidence)[:16],
        "sts_conditioning_text": sts_text(lane, skill, summary, residual),
    }


def sts_row(capsule_row: dict[str, Any]) -> dict[str, Any]:
    text = str(capsule_row.get("sts_conditioning_text") or "")
    return {
        "row_id": "cross_domain_sts_" + stable_hash(capsule_row)[:16],
        "dataset_id": "dataset.cross_domain_sts_capsules.v1",
        "source_type": "cross_domain_sts_capsule",
        "split": "train",
        "capsule_id": capsule_row.get("capsule_id"),
        "lane": capsule_row.get("lane"),
        "skill": capsule_row.get("skill"),
        "prompt": "Condition the active arm/router/decoder on this transferable skill capsule.",
        "answer": text,
        "sts_stream": text,
        "transfer_targets": capsule_row.get("transfer_targets"),
        "source_report": capsule_row.get("source_report"),
        "source_sha256": capsule_row.get("source_sha256"),
        "public_benchmark_training_data_used": False,
        "raw_payload_copied": False,
        "external_inference_calls": 0,
        "created_utc": now(),
    }


def sts_text(lane: str, skill: str, summary: str, residual: str) -> str:
    return (
        f"Cross-domain skill={skill} lane={lane}. "
        f"Lesson: {summary} "
        f"Route residual focus '{residual}' into skeleton choice, tool selection, state memory, and repair planning."
    )


def consumer_contracts_for(targets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for target in targets:
        contract = CONSUMER_CONTRACTS.get(str(target))
        if not contract:
            continue
        consumer = str(contract.get("consumer") or "")
        if not consumer or consumer in seen:
            continue
        seen.add(consumer)
        rows.append(
            {
                "target": str(target),
                "consumer": consumer,
                "control_surface": contract.get("control_surface"),
                "ablation": contract.get("ablation"),
                "evidence_report": contract.get("evidence_report"),
            }
        )
    if not rows:
        fallback = CONSUMER_CONTRACTS["symliquid_state_engine"]
        rows.append(
            {
                "target": "symliquid_state_engine",
                "consumer": fallback["consumer"],
                "control_surface": fallback["control_surface"],
                "ablation": fallback["ablation"],
                "evidence_report": fallback["evidence_report"],
            }
        )
    return rows


def consumer_declared(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    return all(str(row.get(key) or "") for key in ["target", "consumer", "control_surface", "ablation", "evidence_report"])


def causal_evidence() -> dict[str, Any]:
    sts = read_json(REPORTS / "sts_repair_ablation.json", {})
    transfer = read_json(REPORTS / "transfer_eval_suite.json", {})
    decoder = read_json(REPORTS / "decoder_v2_private_ablation_gate.json", {})
    sts_decoder = read_json(REPORTS / "sts_causal_decoder_ablation.json", {})
    symliquid = read_json(REPORTS / "symliquid_state_engine.json", {})

    sts_summary = sts.get("summary") if isinstance(sts.get("summary"), dict) else {}
    transfer_summary = transfer.get("summary") if isinstance(transfer.get("summary"), dict) else {}
    decoder_summary = decoder.get("summary") if isinstance(decoder.get("summary"), dict) else {}
    sts_decoder_summary = sts_decoder.get("summary") if isinstance(sts_decoder.get("summary"), dict) else {}
    sym_weights = symliquid.get("action_kind_weights") if isinstance(symliquid.get("action_kind_weights"), dict) else {}

    sts_delta = number(sts_summary.get("pass_rate_delta"))
    transfer_accuracy = number(transfer_summary.get("accuracy"))
    transfer_task_count = int(number(transfer_summary.get("task_count")))
    sts_candidate_count = int(number(decoder_summary.get("sts_conditioned_candidate_count")))
    sts_rate = number(decoder_summary.get("sts_conditioned_verifier_pass_rate"))
    contract_rate = number(decoder_summary.get("contract_guided_verifier_pass_rate"))
    causal_decoder_private = sts_decoder_summary.get("private") if isinstance(sts_decoder_summary.get("private"), dict) else {}
    causal_decoder_public = sts_decoder_summary.get("public") if isinstance(sts_decoder_summary.get("public"), dict) else {}
    causal_decoder_private_groups = causal_decoder_private.get("groups") if isinstance(causal_decoder_private.get("groups"), dict) else {}
    causal_decoder_public_groups = causal_decoder_public.get("groups") if isinstance(causal_decoder_public.get("groups"), dict) else {}
    causal_decoder_private_sts = (
        causal_decoder_private_groups.get("sts_conditioned")
        if isinstance(causal_decoder_private_groups.get("sts_conditioned"), dict)
        else {}
    )
    causal_decoder_public_sts = (
        causal_decoder_public_groups.get("sts_conditioned")
        if isinstance(causal_decoder_public_groups.get("sts_conditioned"), dict)
        else {}
    )
    causal_decoder_sts_candidate_count = max(
        int(number(causal_decoder_private_sts.get("row_count"))),
        int(number(causal_decoder_public_sts.get("row_count"))),
    )
    causal_decoder_sts_rate = max(
        number(causal_decoder_private_sts.get("verifier_pass_rate")),
        number(causal_decoder_public_sts.get("verifier_pass_rate")),
    )
    causal_decoder_effect = bool(
        sts_decoder.get("trigger_state") == "GREEN"
        and sts_decoder_summary.get("decoder_gate_ready") is True
        and sts_decoder_summary.get("same_seed_non_sts_comparator_present") is True
        and sts_decoder_summary.get("sts_positive_same_seed_lift") is True
        and sts_decoder_summary.get("sts_coverage_non_regressive") is True
        and sts_decoder_summary.get("sts_conditioning_regressed_candidate_coverage") is not True
        and sts_decoder_summary.get("sts_control_contract_ready") is True
        and causal_decoder_sts_candidate_count > 0
        and causal_decoder_sts_rate > 0.0
    )
    route_weight_changed = any(float(number(value)) > 0.0 for value in sym_weights.values())

    measured_transfer_effect = (
        sts_delta > 0.0
        or (transfer_task_count >= 16 and transfer_accuracy >= 0.75)
        or (sts_candidate_count > 0 and sts_rate > 0.0 and sts_rate >= contract_rate)
        or causal_decoder_effect
    )
    return {
        "measured_transfer_effect": bool(measured_transfer_effect),
        "sts_pass_rate_delta": round(sts_delta, 6),
        "transfer_suite_accuracy": round(transfer_accuracy, 6),
        "transfer_suite_task_count": transfer_task_count,
        "decoder_sts_conditioned_candidate_count": sts_candidate_count,
        "decoder_sts_conditioned_verifier_pass_rate": round(sts_rate, 6),
        "decoder_contract_guided_verifier_pass_rate": round(contract_rate, 6),
        "sts_causal_decoder_effect": causal_decoder_effect,
        "sts_causal_decoder_candidate_count": causal_decoder_sts_candidate_count,
        "sts_causal_decoder_verifier_pass_rate": round(causal_decoder_sts_rate, 6),
        "sts_causal_decoder_report": rel_or_abs(REPORTS / "sts_causal_decoder_ablation.json"),
        "symliquid_route_weights_materialized": bool(route_weight_changed),
        "rule": "capsules are mature only after a named consumer changes behavior and a same-seed or baseline comparison shows downstream lift",
    }


def dedupe_capsules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("capsule_id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    out.sort(key=lambda row: (str(row.get("lane")), str(row.get("skill")), str(row.get("capsule_id"))))
    return out


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Cross-Domain STS Capsules",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Capsules: `{summary.get('capsule_count')}`",
        f"- STS rows: `{summary.get('sts_row_count')}`",
        f"- Capsules out: `{summary.get('capsules_out')}`",
        f"- STS out: `{summary.get('sts_out')}`",
        "",
        "## Lanes",
        "",
    ]
    for lane, count in sorted((summary.get("lane_counts") or {}).items()):
        lines.append(f"- `{lane}`: `{count}`")
    lines.extend(["", "## Skills", ""])
    for skill, count in sorted((summary.get("skill_counts") or {}).items()):
        lines.append(f"- `{skill}`: `{count}`")
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}`")
    lines.append("")
    return "\n".join(lines)


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def compact(value: Any) -> Any:
    raw = json.dumps(value, sort_keys=True, default=str)
    if len(raw) <= 1200:
        return value
    return {"sha256": sha256_text(raw), "excerpt": raw[:500], "bytes": len(raw)}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def read_json(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return value if isinstance(value, dict) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    except OSError:
        return []
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
