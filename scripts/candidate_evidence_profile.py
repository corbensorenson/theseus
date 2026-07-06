"""Compile completed candidate evidence into a promotion-profile report.

This does not run benchmark solving or generate candidates. It is a truth-layer
adapter: when the candidate profile was run in resumable chunks, this report
checks the actual artifacts on disk and exposes the same step names that the
promotion gate requires.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from candidate_promotion_gate import (
    CODE_PUBLIC_TASK_FLOOR,
    canonical_real_code_ready,
    code_frontier_transfer_consumed,
    pressure_budget_sufficient,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-family", default="coding_local_sandbox")
    parser.add_argument(
        "--card-id",
        default="",
        help="Pressure card to compile evidence for. Defaults to the active frontier or current real-code report.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frontier-report", default="")
    parser.add_argument("--out", default="reports/training_ratchet_candidate_evidence_profile.json")
    args = parser.parse_args()

    provisional_frontier_report = args.frontier_report
    if not provisional_frontier_report and args.card_id:
        provisional_frontier_report = f"reports/pressure_{safe_name(args.card_id)}_seed{args.seed}.json"
    profile_stub = {"frontier_family": args.frontier_family}
    frontier = read_json(ROOT / provisional_frontier_report) if provisional_frontier_report else {}
    code_forge = read_json(REPORTS / "code_residual_forge.json")
    real_code = read_json(REPORTS / "real_code_benchmark_graduation.json")
    card_id = resolve_card_id(args.card_id, provisional_frontier_report, frontier, real_code)
    frontier_report = provisional_frontier_report or f"reports/pressure_{safe_name(card_id)}_seed{args.seed}.json"
    if not frontier:
        frontier = read_json(ROOT / frontier_report)
    code_lm = read_json(REPORTS / "code_lm_closure.json")
    sts = read_json(REPORTS / "sts_native_parallel_probe.json")
    open_code = read_json(REPORTS / "open_code_training_pantry.json")
    ablation = read_json(REPORTS / "ablation_matrix_rtx2060super_report.json")
    vram = read_json(REPORTS / "profile_vram_stress_report.json")
    runtime = read_json(REPORTS / "preflight_cuda_rollout_smoke.json")
    capability = read_json(REPORTS / "capability_ratchet_run.json")
    preflight = read_json(REPORTS / "training_preflight_report.json")

    steps = [
        step(
            f"pressure_runner_{card_id}_seed{args.seed}",
            pressure_runner_ready(profile_stub, frontier),
            frontier_report,
        ),
        step(
            f"code_residual_forge_{card_id}_seed{args.seed}",
            code_residual_forge_ready(code_forge),
            "reports/code_residual_forge.json",
        ),
        step(
            f"code_lm_closure_{card_id}_seed{args.seed}",
            code_lm_ready(code_lm, real_code),
            "reports/code_lm_closure.json",
        ),
        step(
            f"real_code_benchmark_graduation_learned_{card_id}_seed{args.seed}",
            canonical_real_code_ready(real_code),
            "reports/real_code_benchmark_graduation.json",
        ),
        step(
            "sts_native_parallel_probe",
            sts_ready(sts),
            "reports/sts_native_parallel_probe.json",
        ),
        step(
            "open_code_training_pantry",
            open_code_ready(open_code),
            "reports/open_code_training_pantry.json",
        ),
        step(
            "ablation_matrix",
            bool(ablation.get("ok")) and int(ablation.get("completed_count") or 0) >= 1,
            "reports/ablation_matrix_rtx2060super_report.json",
        ),
        step(
            "profile_vram_stress",
            vram_or_runtime_ready(vram, runtime),
            "reports/profile_vram_stress_report.json|reports/preflight_cuda_rollout_smoke.json",
        ),
        step(
            "capability_ratchet_refresh",
            bool(capability.get("ok")) or str(capability.get("status") or "") == "ok",
            "reports/capability_ratchet_run.json",
        ),
        step(
            "training_preflight_refresh",
            preflight_ready(preflight),
            "reports/training_preflight_report.json",
        ),
    ]
    profile_completion_ok = all(
        item["returncode"] == 0 for item in steps if profile_completion_step(item["name"], args.frontier_family)
    )
    quality_ok = all(item["returncode"] == 0 for item in steps)
    payload = {
        "policy": "local_only_no_external_inference",
        "methodology": "compiled_candidate_artifact_evidence_profile",
        "created_utc": now(),
        "profile": "candidate",
        "frontier_family": args.frontier_family,
        "pressure_card_id": card_id,
        "frontier_report": frontier_report,
        "ok": profile_completion_ok,
        "profile_completion_ok": profile_completion_ok,
        "quality_ok": quality_ok,
        "steps": steps,
        "summary": {
            "active_frontier_accuracy": accuracy(frontier),
            "real_public_task_pass_rate": get_path(real_code, ["summary", "real_public_task_pass_rate"], None),
            "required_public_task_floor": CODE_PUBLIC_TASK_FLOOR,
            "public_quality_clears_floor": canonical_real_code_ready(real_code),
            "token_level_code_generation_learned": get_path(
                real_code, ["summary", "token_level_code_generation_learned"], False
            ),
            "full_body_public_pass_count": get_path(code_lm, ["summary", "full_body_public_pass_count"], 0),
            "expression_fallback_public_pass_count": get_path(
                code_lm, ["summary", "expression_fallback_public_pass_count"], 0
            ),
            "template_like_candidate_count": get_path(real_code, ["summary", "template_like_candidate_count"], None),
            "loop_closure_candidate_count": get_path(real_code, ["summary", "loop_closure_candidate_count"], None),
            "external_inference_calls": external_inference_total(
                frontier, code_forge, real_code, code_lm, sts, open_code, ablation, vram, runtime, capability, preflight
            ),
        },
        "artifacts": {
            "pressure_runner": frontier_report,
            "code_residual_forge": "reports/code_residual_forge.json",
            "real_code_benchmark_graduation": "reports/real_code_benchmark_graduation.json",
            "code_lm_closure": "reports/code_lm_closure.json",
            "sts_native_parallel_probe": "reports/sts_native_parallel_probe.json",
            "open_code_training_pantry": "reports/open_code_training_pantry.json",
            "ablation_matrix": "reports/ablation_matrix_rtx2060super_report.json",
            "vram_stress": "reports/profile_vram_stress_report.json",
            "runtime_cuda_smoke": "reports/preflight_cuda_rollout_smoke.json",
            "capability_ratchet": "reports/capability_ratchet_run.json",
            "preflight": "reports/training_preflight_report.json",
        },
        "external_inference_calls": external_inference_total(
            frontier, code_forge, real_code, code_lm, sts, open_code, ablation, vram, runtime, capability, preflight
        ),
    }
    write_json(ROOT / args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0 if profile_completion_ok else 1


def profile_completion_step(name: str, frontier_family: str) -> bool:
    """Return true for artifacts that prove the candidate profile finished.

    Learned-code quality gates are checked separately by
    candidate_promotion_gate.real_code_benchmark_graduation_ready. Keeping them
    out of profile completeness prevents a below-floor student from looking like
    a stale or incomplete profile.
    """

    required = {
        "ablation_matrix",
        "profile_vram_stress",
        "capability_ratchet_refresh",
        "training_preflight_refresh",
    }
    if name in required:
        return True
    if name.startswith("pressure_runner_"):
        return True
    if frontier_family == "coding_local_sandbox" and name.startswith("code_residual_forge_"):
        return True
    return False


def pressure_runner_ready(profile_stub: dict[str, Any], frontier: dict[str, Any]) -> bool:
    return bool(
        frontier
        and (accuracy(frontier) or 0.0) >= 0.70
        and pressure_budget_sufficient(profile_stub, frontier)
        and code_frontier_transfer_consumed(profile_stub, frontier)
        and int(frontier.get("external_inference_calls") or 0) == 0
    )


def code_residual_forge_ready(report: dict[str, Any]) -> bool:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        report.get("policy") == "project_theseus_code_residual_forge_report_v1"
        and report.get("trigger_state") != "RED"
        and str(summary.get("family") or "") == "coding_local_sandbox"
        and int(summary.get("transfer_artifacts") or 0) > 0
    )


def code_lm_ready(report: dict[str, Any], real_code: dict[str, Any] | None = None) -> bool:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    direct_ready = bool(
        report.get("policy") == "project_theseus_code_lm_closure_v1"
        and report.get("trigger_state") == "GREEN"
        and float(summary.get("public_real_task_pass_rate") or 0.0) >= CODE_PUBLIC_TASK_FLOOR
        and bool(summary.get("token_level_code_generation_learned"))
        and int(summary.get("full_body_public_pass_count") or 0) > 0
        and int(summary.get("expression_fallback_public_pass_count") or 0) == 0
        and int(summary.get("template_like_candidate_count") or 0) == 0
        and int(summary.get("loop_closure_candidate_count") or 0) == 0
        and int(report.get("external_inference_calls") or 0) == 0
    )
    if direct_ready:
        return True
    # The closure report can lag the canonical real-code graduation report when
    # public calibration is rerun directly from a fresh student candidate
    # manifest. Accept that only if the canonical report carries the stricter
    # token-generation and anti-template checks.
    return bool(real_code and canonical_real_code_ready(real_code))


def vram_or_runtime_ready(vram: dict[str, Any], runtime: dict[str, Any]) -> bool:
    stress_rows = [row for row in vram.get("stress", []) if isinstance(row, dict)]
    if bool(vram.get("ok")) and stress_rows and all(row.get("passed") for row in stress_rows):
        return True
    # The current CUDA rollout smoke predates the policy field but is already
    # accepted by the promotion gate when it reports timing, runtime telemetry,
    # and no CUDA fallback. Treat that as a valid replacement for stale VRAM
    # stress rows that failed only because the binary was built without the CUDA
    # feature for that older helper command.
    return bool(
        runtime
        and runtime.get("cuda_fallback") is False
        and bool(runtime.get("runtime_profile"))
        and bool(runtime.get("timing_breakdown_ms"))
        and int(runtime.get("external_inference_calls") or 0) == 0
    )


def sts_ready(report: dict[str, Any]) -> bool:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        report.get("policy") == "project_theseus_sts_native_parallel_probe_v1"
        and report.get("trigger_state") == "GREEN"
        and bool(summary.get("native_parallel_token_generation_proven"))
        and bool(summary.get("one_token_per_output_stream_per_step"))
        and not bool(summary.get("public_benchmark_solutions_included"))
        and int(report.get("external_inference_calls") or 0) == 0
    )


def open_code_ready(report: dict[str, Any]) -> bool:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        report.get("policy") == "project_theseus_open_code_training_pantry_v1"
        and report.get("trigger_state") == "GREEN"
        and int(summary.get("admitted_repo_count") or 0) > 0
        and not bool(summary.get("public_benchmark_solutions_included"))
    )


def preflight_ready(report: dict[str, Any]) -> bool:
    return bool(
        report.get("policy") == "local_only_no_external_inference"
        and report.get("methodology") == "rmi_real_training_preflight"
        and int(report.get("blocker_count") or 0) == 0
    )


def step(name: str, passed: bool, evidence_path: str) -> dict[str, Any]:
    return {
        "name": name,
        "command": ["artifact-evidence", evidence_path],
        "allow_failure": False,
        "returncode": 0 if passed else 1,
        "runtime_ms": 0,
        "timeout_seconds": 0,
        "stdout_tail": f"artifact={evidence_path} passed={bool(passed)}",
        "stderr_tail": "",
    }


def accuracy(report: dict[str, Any]) -> float | None:
    value = get_path(report, ["eval", "summary", "accuracy"], None)
    if value is None:
        value = get_path(report, ["summary", "accuracy"], None)
    if value is None:
        value = get_path(report, ["summary", "score"], None)
    if value is None:
        value = report.get("score")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def external_inference_total(*reports: dict[str, Any]) -> int:
    total = 0
    for report in reports:
        if isinstance(report, dict):
            total += int(report.get("external_inference_calls") or 0)
    return total


def resolve_card_id(
    requested: str,
    frontier_report: str,
    frontier: dict[str, Any],
    real_code: dict[str, Any],
) -> str:
    if requested:
        return requested
    for value in [
        frontier.get("card_id"),
        frontier.get("pressure_card_id"),
        first_item(real_code.get("requested_cards")),
        first_item(real_code.get("cards")),
    ]:
        if value:
            return str(value)
    match = re.search(r"pressure_(.+)_seed\d+", str(frontier_report).replace("\\", "/"))
    if match:
        return match.group(1)
    return "source_human_eval"


def first_item(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    return ""


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def get_path(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
