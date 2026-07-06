#!/usr/bin/env python3
"""Private-only post-v4 fanout precompute policy ablation.

This runner tests whether batched beam precompute is worth keeping for the
current post-v4 private shadow lane. It does not run public calibration and
uses an empty public manifest.
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
REPORTS = ROOT / "reports"
HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "post_v4_private_shadow_transfer_v1_heldout_code_lm_tasks.jsonl"
STS_STREAMS = REPORTS / "post_v4_private_shadow_transfer_v1_private_safe_sts_streams.jsonl"
EMPTY_PUBLIC = REPORTS / "public_safe_broad_transfer_maturity_v4_empty_public.jsonl"
DEFAULT_CHECKPOINT = REPORTS / "student_code_lm_checkpoint_private_residual_repair_v3_private_proof.json"
RELEASE = ROOT / "target" / "release" / ("symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli")
PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"
POST_V4_PUBLIC_RESULT = REPORTS / "real_code_benchmark_graduation_post_v4_seed23_5x32.json"
POST_V4_PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_post_v4_seed23_5x32.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-limit", type=int, default=16)
    parser.add_argument("--candidates-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--timeout-seconds", type=int, default=2)
    parser.add_argument("--checkpoint-in", default=rel(DEFAULT_CHECKPOINT))
    parser.add_argument("--heldout", default=rel(HELDOUT))
    parser.add_argument("--sts-streams", default=rel(STS_STREAMS))
    parser.add_argument("--out", default="reports/post_v4_fanout_precompute_ablation_v1.json")
    parser.add_argument("--markdown-out", default="reports/post_v4_fanout_precompute_ablation_v1.md")
    args = parser.parse_args()

    started = time.time()
    EMPTY_PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    EMPTY_PUBLIC.write_text("", encoding="utf-8")

    runs = [
        run_variant(args, "default", {}),
        run_variant(args, "beam_precompute_off", {"THESEUS_CODE_LM_BATCHED_BEAM_CACHE": "0"}),
    ]
    report = build_report(args, started, runs)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def run_variant(args: argparse.Namespace, variant: str, env: dict[str, str]) -> dict[str, Any]:
    prefix = f"post_v4_precompute_ablation_{variant}_limit{max(1, int(args.task_limit))}"
    private_candidates = REPORTS / f"code_lm_private_candidates_{prefix}.jsonl"
    public_candidates = REPORTS / f"student_code_candidates_{prefix}_unused.jsonl"
    fanout_report = REPORTS / f"code_lm_closure_rust_{prefix}_fanout.json"
    score_report = REPORTS / f"{prefix}_score.json"
    score_md = REPORTS / f"{prefix}_score.md"
    fanout_cmd = [
        rel(RELEASE),
        "generate-code-lm-closure-fanout",
        "--private-curriculum",
        rel(resolve(args.heldout)),
        "--public-task-manifest",
        rel(EMPTY_PUBLIC),
        "--checkpoint-in",
        rel(resolve(args.checkpoint_in)),
        "--seed",
        str(int(args.seed)),
        "--candidates-per-task",
        str(max(1, int(args.candidates_per_task))),
        "--private-candidate-out",
        rel(private_candidates),
        "--public-candidate-out",
        rel(public_candidates),
        "--report-out",
        rel(fanout_report),
        "--public-task-limit",
        "0",
        "--private-eval-limit",
        str(max(1, int(args.task_limit))),
        "--sts-streams",
        rel(resolve(args.sts_streams)),
    ]
    score_cmd = [
        sys.executable,
        "scripts/broad_private_generalization_score_v1.py",
        "--heldout",
        rel(resolve(args.heldout)),
        "--candidates",
        rel(private_candidates),
        "--timeout-seconds",
        str(max(1, int(args.timeout_seconds))),
        "--task-limit",
        str(max(1, int(args.task_limit))),
        "--min-heldout-rows",
        str(max(1, int(args.task_limit))),
        "--out",
        rel(score_report),
        "--markdown-out",
        rel(score_md),
    ]
    fanout_result = run_command(fanout_cmd, env)
    score_result = run_command(score_cmd, {}) if fanout_result["returncode"] == 0 else {"returncode": 99}
    fanout = read_json(fanout_report, {})
    score = read_json(score_report, {})
    public_rows = jsonl_row_count(public_candidates)
    return {
        "variant": variant,
        "env": env,
        "fanout_command": fanout_cmd,
        "score_command": score_cmd,
        "fanout_result": compact_command_result(fanout_result),
        "score_result": compact_command_result(score_result),
        "artifacts": {
            "private_candidates": rel(private_candidates),
            "public_candidates": rel(public_candidates),
            "fanout_report": rel(fanout_report),
            "score_report": rel(score_report),
            "score_markdown": rel(score_md),
        },
        "summary": variant_summary(fanout, score, public_rows),
    }


def variant_summary(fanout: dict[str, Any], score: dict[str, Any], public_rows: int) -> dict[str, Any]:
    fanout_summary = object_field(fanout, "summary")
    score_summary = object_field(score, "summary")
    private_breakdown = object_field(object_field(fanout_summary, "candidate_fanout_runtime_breakdown"), "private")
    phase_categories = object_field(object_field(fanout_summary, "candidate_task_phase_categories"), "private")
    return {
        "fanout_trigger_state": fanout.get("trigger_state"),
        "score_trigger_state": score.get("trigger_state"),
        "runtime_ms": int(first_number(fanout.get("runtime_ms"), 0)),
        "private_candidate_count": int(first_number(fanout_summary.get("private_candidate_count"), 0)),
        "public_candidate_count": int(first_number(fanout_summary.get("public_candidate_count"), public_rows)),
        "public_candidate_rows": public_rows,
        "pass_count": int(first_number(score_summary.get("pass_count"), 0)),
        "task_count": int(first_number(score_summary.get("heldout_task_count"), 0)),
        "pass_rate": float(first_number(score_summary.get("pass_rate"), 0.0)),
        "no_admissible_task_rate": float(first_number(score_summary.get("no_admissible_task_rate"), 1.0)),
        "candidate_generation_and_write_ms": int(first_number(private_breakdown.get("candidate_generation_and_write_ms"), 0)),
        "shared_precompute_wall_ms": int(first_number(private_breakdown.get("shared_precompute_wall_ms"), 0)),
        "per_task_generation_wall_ms": int(first_number(private_breakdown.get("per_task_generation_wall_ms"), 0)),
        "dominant_wall_phase": private_breakdown.get("dominant_wall_phase"),
        "phase_shared_precompute_ms": int(first_number(phase_categories.get("shared_precompute_ms"), 0)),
        "phase_verifier_cache_ms": int(first_number(phase_categories.get("verifier_cache_ms"), 0)),
        "external_inference_calls": int(first_number(fanout.get("external_inference_calls"), score_summary.get("external_inference_calls"), 0)),
    }


def build_report(args: argparse.Namespace, started: float, runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {str(row["variant"]): row for row in runs}
    default = object_field(by_variant.get("default", {}), "summary")
    beam_off = object_field(by_variant.get("beam_precompute_off", {}), "summary")
    default_runtime = int(first_number(default.get("runtime_ms"), 0))
    beam_off_runtime = int(first_number(beam_off.get("runtime_ms"), 0))
    default_pass_rate = float(first_number(default.get("pass_rate"), 0.0))
    beam_off_pass_rate = float(first_number(beam_off.get("pass_rate"), 0.0))
    runtime_delta_ms = default_runtime - beam_off_runtime
    runtime_delta_rate = round(runtime_delta_ms / max(1, default_runtime), 6)
    pass_rate_delta = round(beam_off_pass_rate - default_pass_rate, 6)
    recommendation = "keep_default_beam_precompute"
    if (
        beam_off_pass_rate >= default_pass_rate
        and runtime_delta_rate >= 0.10
        and int(first_number(beam_off.get("public_candidate_rows"), 1)) == 0
    ):
        recommendation = (
            "prefer_beam_precompute_off_for_next_private_scale_probe"
            if int(args.task_limit) >= 64
            else "probe_larger_private_slice_before_policy_change"
        )
    elif int(args.task_limit) < 64:
        recommendation = "probe_larger_private_slice_before_policy_change"
    gates = [
        gate("release_binary_present", RELEASE.exists(), rel(RELEASE)),
        gate("operator_lock_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK)),
        gate("post_v4_public_result_absent", not POST_V4_PUBLIC_RESULT.exists(), rel(POST_V4_PUBLIC_RESULT)),
        gate("post_v4_public_candidates_absent", not POST_V4_PUBLIC_CANDIDATES.exists(), rel(POST_V4_PUBLIC_CANDIDATES)),
        gate("both_variants_ran", all(row["fanout_result"]["returncode"] == 0 and row["score_result"]["returncode"] == 0 for row in runs), [row["variant"] for row in runs]),
        gate("public_candidates_zero", all(int(first_number(object_field(row, "summary").get("public_candidate_rows"), 0)) == 0 for row in runs), [object_field(row, "summary").get("public_candidate_rows") for row in runs]),
        gate("pass_rate_not_regressed", beam_off_pass_rate >= default_pass_rate, {"default": default_pass_rate, "beam_precompute_off": beam_off_pass_rate}),
        gate("external_inference_zero", all(int(first_number(object_field(row, "summary").get("external_inference_calls"), 0)) == 0 for row in runs), 0),
    ]
    hard_failures = [row for row in gates if not row["passed"] and row["gate"] in {"operator_lock_active", "post_v4_public_result_absent", "post_v4_public_candidates_absent", "both_variants_ran", "public_candidates_zero", "external_inference_zero"}]
    trigger_state = "RED" if hard_failures else ("GREEN" if all(row["passed"] for row in gates) else "YELLOW")
    return {
        "policy": "project_theseus_post_v4_fanout_precompute_ablation_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "public_calibration_allowed": False,
        "operator_lock_active": PUBLIC_LOCK.exists(),
        "inputs": {
            "task_limit": max(1, int(args.task_limit)),
            "candidates_per_task": max(1, int(args.candidates_per_task)),
            "seed": int(args.seed),
            "heldout": rel(resolve(args.heldout)),
            "sts_streams": rel(resolve(args.sts_streams)),
            "checkpoint_in": rel(resolve(args.checkpoint_in)),
            "public_tests_used": False,
            "public_solutions_used": False,
        },
        "summary": {
            "recommendation": recommendation,
            "default_runtime_ms": default_runtime,
            "beam_precompute_off_runtime_ms": beam_off_runtime,
            "runtime_delta_ms_default_minus_beam_off": runtime_delta_ms,
            "runtime_delta_rate": runtime_delta_rate,
            "default_pass_rate": default_pass_rate,
            "beam_precompute_off_pass_rate": beam_off_pass_rate,
            "pass_rate_delta": pass_rate_delta,
            "default_shared_precompute_wall_ms": default.get("shared_precompute_wall_ms"),
            "beam_precompute_off_shared_precompute_wall_ms": beam_off.get("shared_precompute_wall_ms"),
            "elapsed_seconds": round(time.time() - started, 3),
            "external_inference_calls": 0,
        },
        "runs": runs,
        "gates": gates,
        "next_actions": next_actions(recommendation),
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def next_actions(recommendation: str) -> list[str]:
    if recommendation == "prefer_beam_precompute_off_for_next_private_scale_probe":
        return [
            "allow the post-v4 autopilot to run the next private runtime probe with THESEUS_CODE_LM_BATCHED_BEAM_CACHE=0",
            "watch the next scale report for pass-rate regression and no-admissible drift",
            "keep public calibration locked",
        ]
    if recommendation == "probe_larger_private_slice_before_policy_change":
        return [
            "rerun this ablation with --task-limit 64 before changing full-scale fanout policy",
            "keep the 1920 private scale run behind runtime profiling",
            "keep public calibration locked",
        ]
    return [
        "keep default batched beam precompute for the next private runtime probe",
        "continue optimizing per-task candidate generation and rank/sort timing",
        "keep public calibration locked",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    lines = [
        "# Post-v4 Fanout Precompute Ablation v1",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Recommendation: `{summary.get('recommendation')}`",
        f"- Default runtime ms: `{summary.get('default_runtime_ms')}`",
        f"- Beam-off runtime ms: `{summary.get('beam_precompute_off_runtime_ms')}`",
        f"- Runtime delta rate: `{summary.get('runtime_delta_rate')}`",
        f"- Default pass rate: `{summary.get('default_pass_rate')}`",
        f"- Beam-off pass rate: `{summary.get('beam_precompute_off_pass_rate')}`",
        f"- Public calibration allowed: `{report.get('public_calibration_allowed')}`",
        "",
        "## Next Actions",
    ]
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def run_command(cmd: list[str], env: dict[str, str]) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env.update(env)
    started = time.time()
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def compact_command_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "returncode": result.get("returncode"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "stdout_tail": result.get("stdout_tail", "")[-500:],
        "stderr_tail": result.get("stderr_tail", "")[-500:],
    }


def jsonl_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def object_field(value: Any, key: str) -> dict[str, Any]:
    field = value.get(key) if isinstance(value, dict) else {}
    return field if isinstance(field, dict) else {}


def first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
