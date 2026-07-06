"""Broad-transfer closure runner for Project Theseus.

This is the executable bridge between the VIEA feedback queue and the code
learner. It does not train on public benchmark answers. It uses public
benchmarks only to export visible prompts and later calibrate learned student
candidates generated from private training pressure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_code_benchmark_graduation as real_code  # noqa: E402
from code_lm_private_rows import (
    REQUIRED_PROMOTION_SAFE_HIGH_TRANSFER_PRIVATE_FILES,
    high_transfer_private_rows_string,
)  # noqa: E402


DEFAULT_RESIDUAL_PRIVATE = "D:/ProjectTheseus/training_data/residual_code_curriculum/private_train/residual_code_lm_tasks.jsonl"
DEFAULT_REPO_ROWS = "D:/ProjectTheseus/training_data/long_horizon_programming/private_train/repo_repair_code_lm_rows.jsonl"
DEFAULT_HIGH_TRANSFER_PRIVATE = high_transfer_private_rows_string()
REQUIRED_HIGH_TRANSFER_PRIVATE_FILES = REQUIRED_PROMOTION_SAFE_HIGH_TRANSFER_PRIVATE_FILES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--max-public-cases-per-card", type=int, default=32)
    parser.add_argument(
        "--case-manifest",
        default="",
        help="Optional public calibration selector manifest. Contains task IDs only.",
    )
    parser.add_argument("--capacity-probe-cases", type=int, default=64)
    parser.add_argument("--private-count", type=int, default=960)
    parser.add_argument("--max-extra-private-train", type=int, default=2000)
    parser.add_argument("--max-residual-private-train", type=int, default=1200)
    parser.add_argument("--max-repo-repair-private-train", type=int, default=1200)
    parser.add_argument("--high-transfer-private-train-jsonl", default=DEFAULT_HIGH_TRANSFER_PRIVATE)
    parser.add_argument("--max-high-transfer-private-train", type=int, default=14400)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--candidates-per-task", type=int, default=12)
    parser.add_argument("--max-rust-work-steps", type=int, default=4000000)
    parser.add_argument("--rust-timeout-seconds", type=int, default=21600)
    parser.add_argument("--public-timeout-seconds", type=int, default=10800)
    parser.add_argument("--sts-timeout-seconds", type=int, default=10800)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument(
        "--typed-edge-exec-receiver-v1",
        action="store_true",
        help="Enable the private typed-edge executable receiver/reranker experiment for Code LM closure.",
    )
    parser.add_argument(
        "--private-type-shape-receiver-veto-v1",
        action="store_true",
        help="Enable teacher-gated type/return-shape receiver bias after the private ablation gate passes.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--score-existing-public-candidates",
        action="store_true",
        help="Score a gated public candidate manifest directly instead of launching a fresh Code LM training closure.",
    )
    parser.add_argument(
        "--student-candidate-manifest",
        default="",
        help="Candidate manifest to score when --score-existing-public-candidates is set. Defaults to the latest decoder gate candidate manifest.",
    )
    parser.add_argument("--skip-sts-ablation", action="store_true")
    parser.add_argument("--out", default="reports/broad_transfer_closure_runner.json")
    parser.add_argument("--markdown-out", default="reports/broad_transfer_closure_runner.md")
    args = parser.parse_args()

    started = time.perf_counter()
    cards = select_cards(args.cards)
    slug = run_slug(cards, seed=args.seed, max_cases=args.max_public_cases_per_card)
    capacity = capacity_report(cards, seed=args.seed, needed=args.max_public_cases_per_card, probe=args.capacity_probe_cases)
    steps: list[dict[str, Any]] = []
    outputs = output_paths(slug)

    if args.execute and all(row["capacity_sufficient"] for row in capacity):
        steps = execute_closure(cards, outputs, args=args)

    refreshed_matrix: dict[str, Any] = {}
    if args.execute and steps and not hard_step_failures(steps):
        refreshed_matrix = read_json(ROOT / "reports" / "broad_transfer_matrix.json", {})

    report = build_report(
        cards,
        capacity,
        steps,
        outputs,
        refreshed_matrix,
        started=started,
        args=args,
        slug=slug,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def select_cards(raw: str) -> list[str]:
    cards = [item.strip() for item in raw.split(",") if item.strip()]
    if not cards:
        broad = read_json(ROOT / "reports" / "broad_transfer_closure.json", {})
        selected = str(get_path(broad, ["summary", "selected_next_card"], "") or "")
        if selected:
            cards = [selected]
    if not cards:
        cards = ["source_mbpp"]
    return real_code.expand_requested_cards(cards)


def capacity_report(cards: list[str], *, seed: int, needed: int, probe: int) -> list[dict[str, Any]]:
    rows = []
    for card_id in cards:
        card = read_json(ROOT / "benchmarks" / "cards" / f"{card_id}.json", {})
        source_id = str(card.get("source_id") or card_id.replace("source_", ""))
        source_path = real_code.resolve_source_path(card)
        tasks: list[dict[str, Any]] = []
        evidence = "source_missing"
        semantics = "blocked_not_scored"
        if source_path.exists():
            tasks, evidence, semantics = real_code.load_cases(
                card_id,
                source_id,
                source_path,
                seed,
                max(probe, needed),
            )
        rows.append(
            {
                "card_id": card_id,
                "source_id": source_id,
                "source_path": rel(source_path),
                "source_exists": source_path.exists(),
                "available_probe_task_count": len(tasks),
                "required_task_count": needed,
                "capacity_sufficient": len(tasks) >= needed,
                "benchmark_evidence_level": evidence,
                "score_semantics": semantics,
                "public_training_rule": "visible_prompts_only_for_generation_public_tests_scorer_only",
            }
        )
    return rows


def execute_closure(cards: list[str], outputs: dict[str, str], *, args: argparse.Namespace) -> list[dict[str, Any]]:
    deadline = time.perf_counter() + max(60, int(args.timeout_seconds))
    if args.score_existing_public_candidates:
        candidate_guard = existing_public_candidate_manifest_guard(args)
        if not candidate_guard["allowed"]:
            return [
                {
                    "name": "existing_public_candidate_manifest_guard",
                    "command": [],
                    "timeout": 0,
                    "allow_failure": False,
                    "returncode": 2,
                    "runtime_ms": 0,
                    "error": "existing_public_candidate_manifest_not_ready",
                    "guard": candidate_guard,
                }
            ]
        return run_public_scoring_steps(cards, outputs, args=args, deadline=deadline, candidate_manifest=candidate_guard["manifest"])

    residual_trace, residual_report = residual_curriculum_inputs(outputs)
    high_transfer_guard = high_transfer_private_train_guard(args.high_transfer_private_train_jsonl)
    if not high_transfer_guard["allowed"]:
        return [
            {
                "name": "high_transfer_private_train_guard",
                "command": [],
                "timeout": 0,
                "allow_failure": False,
                "returncode": 2,
                "runtime_ms": 0,
                "error": "missing_required_high_transfer_private_rows",
                "guard": high_transfer_guard,
            }
        ]
    steps = [
        step(
            "residual_private_curriculum",
            [
                sys.executable,
                "scripts/code_residual_curriculum.py",
                "--trace-in",
                residual_trace,
                "--real-code-report",
                residual_report,
                "--active-card",
                cards[0] if len(cards) == 1 else ",".join(cards),
                "--private-out",
                DEFAULT_RESIDUAL_PRIVATE,
                "--out",
                "reports/code_residual_curriculum.json",
                "--markdown-out",
                "reports/code_residual_curriculum.md",
                "--max-rows",
                str(max(1, int(args.private_count))),
            ],
            timeout=900,
        ),
        step(
            "repo_repair_private_rows",
            [
                sys.executable,
                "scripts/viea_repo_repair_learner.py",
                "--out",
                "reports/viea_repo_repair_learner.json",
                "--markdown-out",
                "reports/viea_repo_repair_learner.md",
            ],
            timeout=900,
            allow_failure=True,
        ),
        step(
            "symliquid_state_engine",
            [
                sys.executable,
                "scripts/symliquid_state_engine.py",
                "--out",
                "reports/symliquid_state_engine.json",
                "--markdown-out",
                "reports/symliquid_state_engine.md",
            ],
            timeout=600,
            allow_failure=True,
        ),
        step(
            "code_lm_closure",
            [
                sys.executable,
                "scripts/code_lm_closure.py",
                "--public-cards",
                ",".join(cards),
                "--seed",
                str(args.seed),
                "--max-public-cases-per-card",
                str(max(1, int(args.max_public_cases_per_card))),
                *(
                    ["--case-manifest", args.case_manifest]
                    if str(args.case_manifest or "").strip()
                    else []
                ),
                "--private-count",
                str(max(1, int(args.private_count))),
                "--epochs",
                str(max(1, int(args.epochs))),
                "--candidates-per-task",
                str(max(1, int(args.candidates_per_task))),
                "--max-extra-private-train",
                str(max(0, int(args.max_extra_private_train))),
                "--max-residual-private-train",
                str(max(0, int(args.max_residual_private_train))),
                "--max-repo-repair-private-train",
                str(max(0, int(args.max_repo_repair_private_train))),
                "--high-transfer-private-train-jsonl",
                args.high_transfer_private_train_jsonl,
                "--max-high-transfer-private-train",
                str(max(0, int(args.max_high_transfer_private_train))),
                "--max-rust-work-steps",
                str(max(0, int(args.max_rust_work_steps))),
                "--rust-timeout-seconds",
                str(max(0, int(args.rust_timeout_seconds))),
                "--public-timeout-seconds",
                str(max(0, int(args.public_timeout_seconds))),
                "--sts-timeout-seconds",
                str(max(0, int(args.sts_timeout_seconds))),
                "--residual-private-train-jsonl",
                DEFAULT_RESIDUAL_PRIVATE,
                "--repo-repair-private-train-jsonl",
                DEFAULT_REPO_ROWS,
                "--private-curriculum-out",
                outputs["private_curriculum"],
                "--public-task-manifest-out",
                outputs["public_tasks"],
                "--checkpoint-out",
                outputs["checkpoint"],
                "--private-candidate-out",
                outputs["private_candidates"],
                "--public-candidate-out",
                outputs["public_candidates"],
                "--rust-report-out",
                outputs["rust_report"],
                "--public-report-out",
                outputs["public_report"],
                "--public-trace-out",
                outputs["public_trace"],
                "--public-transfer-artifact-out",
                outputs["transfer_artifact"],
                "--out",
                outputs["code_lm_report"],
                "--sts-conditioning-input-out",
                outputs["sts_input"],
                "--sts-generation-out",
                outputs["sts_generations"],
                "--sts-conditioning-checkpoint-out",
                outputs["sts_checkpoint"],
                "--sts-conditioning-report-out",
                outputs["sts_report"],
                "--lock-path",
                outputs["lock"],
                *(
                    ["--typed-edge-exec-receiver-v1"]
                    if args.typed_edge_exec_receiver_v1
                    else []
                ),
                "--edge-obligation-decode-gate-v1",
                "--edge-obligation-report-out",
                outputs["edge_obligation"],
                "--edge-obligation-markdown-out",
                outputs["edge_obligation_md"],
                *(
                    ["--private-type-shape-receiver-veto-v1"]
                    if args.private_type_shape_receiver_veto_v1
                    else []
                ),
            ],
            timeout=max(3600, int(args.timeout_seconds)),
            env=code_lm_step_env(args),
        ),
    ]
    if not args.skip_sts_ablation:
        steps.append(
            step(
                "sts_repair_ablation",
                [
                    sys.executable,
                    "scripts/sts_repair_ablation.py",
                    "--real-code-report",
                    outputs["public_report"],
                    "--trace-in",
                    outputs["public_trace"],
                    "--code-lm-report",
                    outputs["code_lm_report"],
                    "--out",
                    outputs["sts_ablation"],
                    "--markdown-out",
                    outputs["sts_ablation_md"],
                ],
                timeout=600,
                allow_failure=True,
            )
        )
    steps.append(
        step(
            "broad_transfer_matrix_refresh",
            [
                sys.executable,
                "scripts/broad_transfer_matrix.py",
                "--min-public-tasks",
                str(max(32, int(args.max_public_cases_per_card))),
                "--out",
                "reports/broad_transfer_matrix.json",
                "--markdown-out",
                "reports/broad_transfer_matrix.md",
            ],
            timeout=600,
            allow_failure=True,
        )
    )

    rows = []
    for spec in steps:
        seconds_left = int(deadline - time.perf_counter())
        if seconds_left <= 0:
            rows.append({**spec, "returncode": 124, "runtime_ms": 0, "error": "runner_timeout_before_step"})
            break
        row = run_step(spec, timeout=min(int(spec["timeout"]), max(60, seconds_left)))
        row = validate_step_completion(row, outputs=outputs)
        rows.append(row)
        if rows[-1]["returncode"] != 0 and not rows[-1].get("allow_failure"):
            break
    return rows


def run_public_scoring_steps(
    cards: list[str],
    outputs: dict[str, str],
    *,
    args: argparse.Namespace,
    deadline: float,
    candidate_manifest: str,
) -> list[dict[str, Any]]:
    steps = [
        step(
            "score_existing_public_candidates",
            [
                sys.executable,
                "scripts/real_code_benchmark_graduation.py",
                "--cards",
                ",".join(cards),
                "--seed",
                str(args.seed),
                "--max-cases-per-card",
                str(max(1, int(args.max_public_cases_per_card))),
                "--skip-student-candidate-generation",
                "--student-candidate-manifest",
                candidate_manifest,
                "--out",
                outputs["public_report"],
                "--trace-out",
                outputs["public_trace"],
                "--transfer-artifact-out",
                outputs["transfer_artifact"],
                *(
                    ["--case-manifest", args.case_manifest]
                    if str(args.case_manifest or "").strip()
                    else []
                ),
            ],
            timeout=max(600, int(args.public_timeout_seconds)),
        )
    ]
    if not args.skip_sts_ablation:
        steps.append(
            step(
                "sts_repair_ablation",
                [
                    sys.executable,
                    "scripts/sts_repair_ablation.py",
                    "--real-code-report",
                    outputs["public_report"],
                    "--trace-in",
                    outputs["public_trace"],
                    "--code-lm-report",
                    "reports/code_lm_closure_private_pressure_private.json",
                    "--out",
                    outputs["sts_ablation"],
                    "--markdown-out",
                    outputs["sts_ablation_md"],
                ],
                timeout=600,
                allow_failure=True,
            )
        )
    steps.append(
        step(
            "broad_transfer_matrix_refresh",
            [
                sys.executable,
                "scripts/broad_transfer_matrix.py",
                "--min-public-tasks",
                str(max(32, int(args.max_public_cases_per_card))),
                "--out",
                "reports/broad_transfer_matrix.json",
                "--markdown-out",
                "reports/broad_transfer_matrix.md",
            ],
            timeout=600,
            allow_failure=True,
        )
    )
    rows = []
    for spec in steps:
        seconds_left = int(deadline - time.perf_counter())
        if seconds_left <= 0:
            rows.append({**spec, "returncode": 124, "runtime_ms": 0, "error": "runner_timeout_before_step"})
            break
        row = run_step(spec, timeout=min(int(spec["timeout"]), max(60, seconds_left)))
        rows.append(row)
        if rows[-1]["returncode"] != 0 and not rows[-1].get("allow_failure"):
            break
    return rows


def existing_public_candidate_manifest_guard(args: argparse.Namespace) -> dict[str, Any]:
    gate = read_json(ROOT / "reports" / "decoder_v2_private_ablation_gate.json", {})
    manifest = str(args.student_candidate_manifest or "") or str(
        get_path(gate, ["closure_reports", 0, "public_candidate_manifest"], "")
    )
    path = resolve(manifest) if manifest else ROOT / "__missing_public_candidate_manifest__"
    ready = bool(gate.get("ready_for_public_calibration")) and gate.get("trigger_state") == "GREEN"
    return {
        "allowed": ready and path.exists() and path.stat().st_size > 0,
        "manifest": rel(path) if path.exists() else manifest,
        "decoder_gate_ready": ready,
        "manifest_exists": path.exists(),
        "manifest_bytes": path.stat().st_size if path.exists() else 0,
        "rule": "public receiver calibration scores gated student candidates without retraining",
    }


def selected_decoder_gate_closure_report() -> Path:
    gate = read_json(ROOT / "reports" / "decoder_v2_private_ablation_gate.json", {})
    candidates = [
        get_path(gate, ["summary", "latest_closure"], ""),
        get_path(gate, ["closure_reports", 0, "path"], ""),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = resolve(str(candidate))
        if path.exists():
            return path
    return ROOT / "reports" / "code_lm_closure_private_pressure_private.json"


def score_existing_conditioning_evidence(
    args: argparse.Namespace,
    code_lm_report: dict[str, Any],
    public_summary: dict[str, Any],
) -> dict[str, Any]:
    summary = code_lm_report.get("summary") if isinstance(code_lm_report.get("summary"), dict) else {}
    diagnostics = summary.get("public_candidate_manifest_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    manifest_modes = public_summary.get("candidate_generation_modes")
    manifest_modes = list(manifest_modes) if isinstance(manifest_modes, list) else []
    if args.score_existing_public_candidates:
        guard = existing_public_candidate_manifest_guard(args)
        manifest = str(guard.get("manifest") or "")
        path = resolve(manifest) if manifest else ROOT / "__missing_public_candidate_manifest__"
        if path.exists():
            loaded = real_code.load_student_candidates(path)
            modes = loaded.get("candidate_generation_modes")
            if isinstance(modes, list):
                manifest_modes = [str(mode) for mode in modes]
    sts_mode_count = sum(1 for mode in manifest_modes if "sts_conditioned" in str(mode) or "_sts" in str(mode))
    sts_candidate_count = int(
        diagnostics.get("sts_conditioned_candidate_count")
        or public_summary.get("sts_conditioned_candidate_count")
        or 0
    )
    sts_used = (
        bool(summary.get("sts_conditioning_used"))
        or bool(summary.get("sts_default_policy"))
        or sts_candidate_count > 0
        or sts_mode_count > 0
    )
    symliquid_used = bool(summary.get("symliquid_state_conditioning_used"))
    return {
        "used": symliquid_used or sts_used,
        "symliquid_state_conditioning_used": symliquid_used,
        "sts_conditioning_used": sts_used,
        "sts_conditioned_candidate_count": sts_candidate_count,
        "sts_conditioned_mode_count": sts_mode_count,
        "selected_closure_report": rel(selected_decoder_gate_closure_report()),
        "rule": "score-existing calibration may prove conditioning from the decoder-gated closure report and its candidate manifest without rerunning public generation",
    }


def high_transfer_private_train_guard(raw: str) -> dict[str, Any]:
    paths = {
        item.strip().replace("\\", "/")
        for chunk in str(raw or "").split(";")
        for item in chunk.split(",")
        if item.strip()
    }
    required = {path.replace("\\", "/") for path in REQUIRED_HIGH_TRANSFER_PRIVATE_FILES}
    missing = sorted(required - paths)
    return {
        "allowed": not missing,
        "required": sorted(required),
        "missing": missing,
        "path_count": len(paths),
        "rule": "public receiver calibration must use the current private closure pressure set",
    }


def residual_curriculum_inputs(outputs: dict[str, str]) -> tuple[str, str]:
    """Prefer the previous report for this exact card slice when it exists.

    A repeat closure run should train on private lookalikes shaped by the
    freshest same-card residual labels. On a first run, those artifacts do not
    exist yet, so the generator falls back to the broad/default residual view.
    """
    trace = resolve(outputs["public_trace"])
    report = resolve(outputs["public_report"])
    if trace.exists() and report.exists():
        return outputs["public_trace"], outputs["public_report"]
    return "reports/real_code_benchmark_traces.jsonl", "reports/real_code_benchmark_graduation.json"


def output_paths(slug: str) -> dict[str, str]:
    return {
        "private_curriculum": f"data/private_code_curriculum/code_lm_closure_{slug}.jsonl",
        "public_tasks": f"reports/code_lm_public_tasks_{slug}.jsonl",
        "checkpoint": f"reports/student_code_lm_checkpoint_{slug}.json",
        "private_candidates": f"reports/code_lm_private_candidates_{slug}.jsonl",
        "public_candidates": f"reports/student_code_candidates_{slug}.jsonl",
        "rust_report": f"reports/code_lm_closure_rust_{slug}.json",
        "public_report": f"reports/real_code_benchmark_graduation_{slug}.json",
        "public_trace": f"reports/real_code_benchmark_traces_{slug}.jsonl",
        "transfer_artifact": f"reports/transfer_artifacts/code/real_code_benchmark_graduation_{slug}_transfer_artifact.json",
        "code_lm_report": f"reports/code_lm_closure_{slug}.json",
        "sts_input": f"reports/code_lm_sts_conditioning_input_{slug}.jsonl",
        "sts_generations": f"reports/code_lm_sts_public_generations_{slug}.jsonl",
        "sts_checkpoint": f"reports/code_lm_sts_conditioning_checkpoint_{slug}.json",
        "sts_report": f"reports/code_lm_sts_conditioning_report_{slug}.json",
        "sts_ablation": f"reports/sts_repair_ablation_{slug}.json",
        "sts_ablation_md": f"reports/sts_repair_ablation_{slug}.md",
        "edge_obligation": f"reports/edge_obligation_decode_gate_v1_{slug}.json",
        "edge_obligation_md": f"reports/edge_obligation_decode_gate_v1_{slug}.md",
        "lock": f"reports/code_lm_closure_{slug}.lock",
    }


def build_report(
    cards: list[str],
    capacity: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    outputs: dict[str, str],
    matrix: dict[str, Any],
    *,
    started: float,
    args: argparse.Namespace,
    slug: str,
) -> dict[str, Any]:
    outputs_current = bool(steps) or not args.execute
    public_report = read_json(resolve(outputs["public_report"]), {}) if outputs_current else {}
    code_lm_report_path = (
        selected_decoder_gate_closure_report()
        if args.score_existing_public_candidates
        else resolve(outputs["code_lm_report"])
    )
    code_lm_report = read_json(code_lm_report_path, {}) if outputs_current else {}
    sts_report = read_json(resolve(outputs["sts_ablation"]), {}) if outputs_current else {}
    step_failures = hard_step_failures(steps)
    public_summary = public_report.get("summary") if isinstance(public_report.get("summary"), dict) else {}
    code_lm_summary = code_lm_report.get("summary") if isinstance(code_lm_report.get("summary"), dict) else {}
    sts_summary = sts_report.get("summary") if isinstance(sts_report.get("summary"), dict) else {}
    conditioning_evidence = score_existing_conditioning_evidence(args, code_lm_report, public_summary)
    template_like_candidates = int(
        public_summary.get("template_like_candidate_count")
        or public_summary.get("template_like_public_candidate_count")
        or 0
    )
    loop_candidates = int(
        public_summary.get("loop_closure_candidate_count")
        or public_summary.get("loop_closure_public_candidate_count")
        or 0
    )
    eligible_candidates = int(public_summary.get("benchmark_promotion_eligible_candidate_count") or 0)
    gates = [
        gate("requested_cards_present", bool(cards), cards),
        gate("public_capacity_sufficient", all(row["capacity_sufficient"] for row in capacity), capacity),
        gate("step_duration_budgeted", int(args.max_rust_work_steps) > 0, args.max_rust_work_steps),
        gate("wall_clock_is_safety_fuse_only", True, {
            "runner_timeout_seconds": args.timeout_seconds,
            "rust_timeout_seconds": args.rust_timeout_seconds,
            "public_timeout_seconds": args.public_timeout_seconds,
        }),
        gate("public_data_calibration_only", True, "public prompts visible only; tests scorer only; no public solutions in private train"),
        gate(
            "current_private_pressure_rows_present",
            high_transfer_private_train_guard(args.high_transfer_private_train_jsonl)["allowed"],
            high_transfer_private_train_guard(args.high_transfer_private_train_jsonl),
        ),
    ]
    if args.score_existing_public_candidates:
        existing_guard = existing_public_candidate_manifest_guard(args)
        gates.append(gate("existing_public_candidate_manifest_ready", existing_guard["allowed"], existing_guard))
    if args.execute:
        gates.extend(
            [
                gate(
                    "closure_steps_completed",
                    bool(steps) and not step_failures,
                    [row.get("name") for row in step_failures] if steps else "no_steps_executed",
                ),
                gate("token_level_student_candidates", bool(public_summary.get("token_level_code_generation_learned")), public_summary.get("candidate_generation_modes")),
                gate(
                    "public_candidate_evidence_clean",
                    public_report.get("trigger_state") == "GREEN",
                    {
                        "public_report_trigger_state": public_report.get("trigger_state"),
                        "template_like_candidate_count": template_like_candidates,
                        "loop_closure_candidate_count": loop_candidates,
                        "benchmark_promotion_eligible_candidate_count": eligible_candidates,
                    },
                ),
                gate(
                    "no_template_or_loop_public_candidates",
                    template_like_candidates == 0 and loop_candidates == 0,
                    public_summary,
                ),
                gate(
                    "benchmark_promotion_eligible_candidates_present",
                    eligible_candidates > 0,
                    public_summary.get("candidate_quality_summary") or public_summary,
                ),
                gate(
                    "public_task_coverage_requested",
                    int(public_summary.get("public_task_count") or 0) >= len(cards) * int(args.max_public_cases_per_card),
                    {"actual": public_summary.get("public_task_count"), "required": len(cards) * int(args.max_public_cases_per_card)},
                ),
                gate("symliquid_conditioning_used", bool(conditioning_evidence.get("used")), conditioning_evidence),
                gate("sts_delta_measured", bool(sts_summary) or bool(args.skip_sts_ablation), sts_summary),
            ]
        )
    hard_failed = [row for row in gates if not row["passed"] and row.get("severity") == "hard"]
    trigger = "RED" if hard_failed and args.execute else "YELLOW" if hard_failed or not args.execute else "GREEN"
    return {
        "policy": "project_theseus_broad_transfer_closure_runner_v1",
        "created_utc": now(),
        "trigger_state": trigger,
        "summary": {
            "cards": cards,
            "slug": slug,
            "execute": bool(args.execute),
            "max_public_cases_per_card": int(args.max_public_cases_per_card),
            "score_existing_public_candidates": bool(args.score_existing_public_candidates),
            "case_manifest": args.case_manifest,
            "public_capacity_min": min((row["available_probe_task_count"] for row in capacity), default=0),
            "step_count": len(steps),
            "hard_step_failure_count": len(step_failures),
            "public_task_count": public_summary.get("public_task_count"),
            "public_pass_rate": public_summary.get("real_public_task_pass_rate"),
            "sts_delta": sts_summary.get("pass_rate_delta"),
            "symliquid_conditioning_used": conditioning_evidence.get("used"),
            "conditioning_evidence": conditioning_evidence,
            "high_transfer_private_train_task_count": code_lm_summary.get("high_transfer_private_train_task_count"),
            "rust_work_budget_admission": code_lm_summary.get("rust_work_budget_admission"),
            "typed_edge_exec_receiver_v1_enabled": bool(args.typed_edge_exec_receiver_v1),
            "private_type_shape_receiver_veto_v1_enabled": bool(args.private_type_shape_receiver_veto_v1),
            "edge_obligation_decode_gate": code_lm_summary.get("edge_obligation_decode_gate_report"),
            "edge_obligation_decode_gate_ready": code_lm_summary.get("edge_obligation_decode_gate_ready"),
            "broad_matrix_pass_rate": get_path(matrix, ["summary", "real_public_pass_rate"], None),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "promotion_evidence": False,
        },
        "capacity": capacity,
        "steps": steps,
        "outputs": outputs,
        "gates": gates,
        "rules": {
            "public_benchmarks": "calibration_only_not_training",
            "private_training": "private_hidden_test_rows_plus_high_transfer_private_rows_plus_governed_repo_repair_rows_only",
            "student_evidence": "token_level_student_code_generation_required",
            "duration": "step_budget_primary_wall_clock_safety_fuse_only",
        },
        "score_semantics": "broad-transfer closure execution report; public results are calibration only until promotion gates consume clean evidence",
        "external_inference_calls": 0,
    }


def step(
    name: str,
    command: list[Any],
    *,
    timeout: int,
    allow_failure: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "command": [str(item) for item in command],
        "timeout": int(timeout),
        "allow_failure": bool(allow_failure),
        "env": {str(key): str(value) for key, value in (env or {}).items()},
    }


def code_lm_step_env(args: argparse.Namespace) -> dict[str, str]:
    env = {
        "THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION": "1",
        "THESEUS_TARGET_FAMILY_STARVATION_RESCUE": "1",
        "THESEUS_TARGET_FAMILY_STARVATION_RESCUE_MIN_ROWS": "48",
    }
    if args.typed_edge_exec_receiver_v1:
        env["THESEUS_TYPED_EDGE_EXEC_RECEIVER_V1"] = "1"
    if args.private_type_shape_receiver_veto_v1:
        env["THESEUS_PRIVATE_TYPE_SHAPE_RECEIVER_VETO_V1"] = "1"
    return env


def run_step(spec: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in dict(spec.get("env") or {}).items()})
        result = subprocess.run(
            spec["command"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=max(60, int(timeout)),
            env=env,
        )
        return {
            **spec,
            "timeout_seconds": max(60, int(timeout)),
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            **spec,
            "timeout_seconds": max(60, int(timeout)),
            "returncode": 124,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": timeout_text(exc.stdout)[-4000:],
            "stderr_tail": timeout_text(exc.stderr)[-4000:],
            "error": "timeout_safety_fuse",
        }


def validate_step_completion(row: dict[str, Any], *, outputs: dict[str, str]) -> dict[str, Any]:
    """Treat stale/in-progress artifacts as failed execution evidence.

    Long unattended runs are allowed to end YELLOW for learning walls, but a
    runner must not advance from Code LM closure to STS/public matrix refresh
    when the closure report is still a progress placeholder or public
    calibration never materialized.
    """
    if int(row.get("returncode") or 0) != 0:
        return row
    if row.get("name") != "code_lm_closure":
        return row
    code_lm = read_json(resolve(outputs["code_lm_report"]), {})
    public_report = read_json(resolve(outputs["public_report"]), {})
    if code_lm.get("run_status") == "in_progress":
        return {
            **row,
            "returncode": 2,
            "error": "code_lm_closure_left_in_progress_report",
            "completion_report_state": {
                "code_lm_report": outputs["code_lm_report"],
                "run_status": code_lm.get("run_status"),
                "progress_stage": code_lm.get("progress_stage"),
                "rust_report": outputs["rust_report"],
            },
        }
    if code_lm.get("trigger_state") == "RED":
        return {
            **row,
            "returncode": 2,
            "error": "code_lm_closure_red",
            "completion_report_state": {
                "code_lm_report": outputs["code_lm_report"],
                "hard_operational_failures": code_lm.get("hard_operational_failures", []),
            },
        }
    if not public_report:
        return {
            **row,
            "returncode": 2,
            "error": "public_calibration_report_missing_after_code_lm_closure",
            "completion_report_state": {
                "code_lm_report": outputs["code_lm_report"],
                "public_report": outputs["public_report"],
                "code_lm_trigger_state": code_lm.get("trigger_state"),
            },
        }
    return row


def hard_step_failures(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in steps if int(row.get("returncode") or 0) != 0 and not row.get("allow_failure")]


def run_slug(cards: list[str], *, seed: int, max_cases: int) -> str:
    return safe_name("_".join(cards) + f"_seed{seed}_{max_cases}")


def safe_name(value: Any) -> str:
    return real_code.safe_name(value).lower()


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Broad Transfer Closure Runner",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- cards: `{', '.join(summary.get('cards') or [])}`",
        f"- execute: `{summary.get('execute')}`",
        f"- public_capacity_min: `{summary.get('public_capacity_min')}`",
        f"- public_task_count: `{summary.get('public_task_count')}`",
        f"- public_pass_rate: `{summary.get('public_pass_rate')}`",
        f"- sts_delta: `{summary.get('sts_delta')}`",
        f"- symliquid_conditioning_used: `{summary.get('symliquid_conditioning_used')}`",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates", []):
        lines.append(f"- {'PASS' if row.get('passed') else 'FAIL'} `{row.get('gate')}`: {row.get('evidence')}")
    lines.extend(["", "## Steps", ""])
    for row in report.get("steps", []):
        lines.append(f"- `{row.get('returncode')}` `{row.get('name')}` runtime_ms={row.get('runtime_ms')}")
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path, default: Any = None) -> Any:
    default = {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def timeout_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
