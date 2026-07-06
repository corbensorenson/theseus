#!/usr/bin/env python3
"""Gate Broad Private Generalization Ladder v1 evidence.

The gate is intentionally private-only. A GREEN state means the broad private
heldout floor cleared. A YELLOW state with completion_evidence_status
``precise_blocker`` is also useful: it means an unattended run produced enough
evidence to know the next repair target without spending another public
calibration.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from readiness_freshness import freshness_report


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_CURRICULUM = REPORTS / "broad_private_generalization_ladder_v1.json"
DEFAULT_SCORE = REPORTS / "broad_private_generalization_score_v1.json"
DEFAULT_UNATTENDED = REPORTS / "broad_private_generalization_unattended_v1.json"
DEFAULT_LOCK = REPORTS / "public_calibration_operator_lock.flag"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", default=rel(DEFAULT_CURRICULUM))
    parser.add_argument("--score", default=rel(DEFAULT_SCORE))
    parser.add_argument("--unattended", default=rel(DEFAULT_UNATTENDED))
    parser.add_argument("--operator-lock", default=rel(DEFAULT_LOCK))
    parser.add_argument("--out", default="reports/broad_private_generalization_gate_v1.json")
    parser.add_argument("--markdown-out", default="reports/broad_private_generalization_gate_v1.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    curriculum_path = resolve(args.curriculum)
    score_path = resolve(args.score)
    unattended_path = resolve(args.unattended)
    lock_path = resolve(args.operator_lock)

    curriculum = read_json(curriculum_path, {})
    score = read_json(score_path, {})
    unattended = read_json(unattended_path, {})
    score_summary = score.get("summary") if isinstance(score.get("summary"), dict) else {}
    curriculum_summary = curriculum.get("summary") if isinstance(curriculum.get("summary"), dict) else {}
    unattended_summary = unattended.get("summary") if isinstance(unattended.get("summary"), dict) else {}
    precise_blocker = first_precise_blocker(score, unattended)
    current_evidence = current_decoder_evidence(score_path, score, unattended_path)

    gates = [
        gate("public_calibration_operator_lock_active", lock_path.exists(), public_lock_evidence(lock_path)),
        gate("curriculum_report_present", curriculum_path.exists(), rel(curriculum_path)),
        gate("curriculum_green", curriculum.get("trigger_state") == "GREEN", curriculum.get("trigger_state")),
        gate(
            "private_train_rows_ge_2400",
            int(curriculum_summary.get("private_train_row_count") or 0) >= 2400,
            curriculum_summary.get("private_train_row_count"),
        ),
        gate(
            "private_heldout_rows_ge_1000",
            int(curriculum_summary.get("private_heldout_row_count") or 0) >= 1000,
            curriculum_summary.get("private_heldout_row_count"),
        ),
        gate(
            "curriculum_solution_failures_zero",
            int(curriculum_summary.get("private_train_solution_failures") or 0) == 0
            and int(curriculum_summary.get("private_heldout_solution_failures") or 0) == 0,
            {
                "train": curriculum_summary.get("private_train_solution_failures"),
                "heldout": curriculum_summary.get("private_heldout_solution_failures"),
            },
        ),
        gate(
            "curriculum_public_data_leakage_zero",
            int(curriculum_summary.get("public_data_leakage_hit_count") or 0) == 0,
            curriculum_summary.get("public_data_leakage_hit_count"),
        ),
        gate("unattended_report_present", unattended_path.exists(), rel(unattended_path)),
        gate(
            "unattended_not_red",
            unattended.get("trigger_state") in {"GREEN", "YELLOW"},
            {
                "state": unattended.get("trigger_state"),
                "completion_evidence_status": unattended_summary.get("completion_evidence_status"),
            },
        ),
        gate("score_report_present", score_path.exists(), rel(score_path)),
        gate("score_and_candidate_artifacts_current_for_decoder_source_and_release", current_evidence["fresh"], current_evidence),
        gate(
            "broad_private_pass_rate_floor",
            float(score_summary.get("pass_rate") or 0.0) >= 0.70,
            {"observed": score_summary.get("pass_rate"), "minimum": 0.70},
        ),
        gate(
            "no_admissible_rate_floor",
            numeric(score_summary.get("no_admissible_task_rate"), 1.0) <= 0.03,
            {"observed": score_summary.get("no_admissible_task_rate"), "maximum": 0.03},
        ),
        gate(
            "sts_same_seed_positive",
            float(score_summary.get("sts_delta") or 0.0) > 0.0,
            {"delta": score_summary.get("sts_delta")},
        ),
        gate(
            "sts_regressions_zero",
            int(score_summary.get("sts_regressions") or 0) == 0,
            {"regressions": score_summary.get("sts_regressions")},
        ),
        gate(
            "score_public_data_leakage_zero",
            int(score_summary.get("public_data_leakage_hit_count") or 0) == 0,
            score_summary.get("public_data_leakage_hit_count"),
        ),
    ]

    hard_failures = [
        row
        for row in gates
        if not row["passed"]
        and row["gate"]
        in {
            "public_calibration_operator_lock_active",
            "curriculum_report_present",
            "curriculum_green",
            "curriculum_solution_failures_zero",
            "curriculum_public_data_leakage_zero",
            "unattended_report_present",
            "unattended_not_red",
            "score_public_data_leakage_zero",
        }
    ]
    all_passed = all(row["passed"] for row in gates)
    if all_passed:
        trigger_state = "GREEN"
        completion_evidence_status = "green_transfer"
    elif hard_failures:
        trigger_state = "RED"
        completion_evidence_status = "hard_blocker"
    elif precise_blocker:
        trigger_state = "YELLOW"
        completion_evidence_status = "precise_blocker"
    else:
        trigger_state = "YELLOW"
        completion_evidence_status = "pending_score_or_repair"

    blockers = [row for row in gates if not row["passed"]]
    return {
        "policy": "project_theseus_broad_private_generalization_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "curriculum": rel(curriculum_path),
            "score": rel(score_path),
            "unattended": rel(unattended_path),
            "operator_lock": rel(lock_path),
            "public_calibration": "locked",
        },
        "summary": {
            "completion_evidence_status": completion_evidence_status,
            "private_train_rows": curriculum_summary.get("private_train_row_count"),
            "private_heldout_rows": curriculum_summary.get("private_heldout_row_count"),
            "heldout_pass_rate": score_summary.get("pass_rate"),
            "heldout_passes": score_summary.get("pass_count"),
            "heldout_task_count": score_summary.get("heldout_task_count"),
            "no_admissible_task_rate": score_summary.get("no_admissible_task_rate"),
            "sts_delta": score_summary.get("sts_delta"),
            "sts_regressions": score_summary.get("sts_regressions"),
            "unattended_completion": unattended_summary.get("completion_evidence_status"),
            "decoder_source_release_fresh": current_evidence["fresh"],
            "decoder_source_release_stale_reasons": current_evidence["stale_reasons"],
            "precise_blocker": precise_blocker,
            "blocker_count": len(blockers),
        },
        "gates": gates,
        "blockers": blockers,
        "next_actions": next_actions(trigger_state, score, unattended, blockers, precise_blocker, current_evidence),
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def current_decoder_evidence(score_path: Path, score: dict[str, Any], unattended_path: Path) -> dict[str, Any]:
    inputs = score.get("inputs") if isinstance(score.get("inputs"), dict) else {}
    artifacts = {
        "score_report": score_path,
        "unattended_report": unattended_path,
    }
    for key in ["candidates", "control_candidates"]:
        value = str(inputs.get(key) or "").strip()
        if value:
            artifacts[key] = resolve(value)
    return freshness_report(
        artifacts,
        root=ROOT,
        rule=(
            "broad-private score evidence must be regenerated after decoder source "
            "changes or after rebuilding target/release/symliquid-cli"
        ),
    )


def first_precise_blocker(score: dict[str, Any], unattended: dict[str, Any]) -> dict[str, Any]:
    if isinstance(unattended.get("precise_blocker"), dict) and unattended.get("precise_blocker"):
        return unattended["precise_blocker"]
    summary = score.get("summary") if isinstance(score.get("summary"), dict) else {}
    if not summary:
        return {}
    if int(summary.get("candidate_row_count") or 0) <= 0:
        return {"kind": "candidate_generation_missing", "detail": "no candidate rows were produced for broad private heldout"}
    if float(summary.get("no_admissible_task_rate") or 0.0) > 0.03:
        return {
            "kind": "candidate_coverage",
            "detail": "no-admissible rate exceeds broad private floor",
            "observed": summary.get("no_admissible_task_rate"),
        }
    if float(summary.get("pass_rate") or 0.0) < 0.70:
        return {
            "kind": "broad_private_transfer_floor",
            "detail": "candidate rows exist but broad private heldout pass rate is below floor",
            "observed": summary.get("pass_rate"),
            "weakest_families": summary.get("weakest_families"),
        }
    if float(summary.get("sts_delta") or 0.0) <= 0.0 or int(summary.get("sts_regressions") or 0) > 0:
        return {
            "kind": "sts_causal_control",
            "detail": "STS-on did not beat STS-off cleanly on same private heldout",
            "delta": summary.get("sts_delta"),
            "regressions": summary.get("sts_regressions"),
        }
    return {}


def numeric(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def next_actions(
    trigger_state: str,
    score: dict[str, Any],
    unattended: dict[str, Any],
    blockers: list[dict[str, Any]],
    precise_blocker: dict[str, Any],
    current_evidence: dict[str, Any],
) -> list[str]:
    if trigger_state == "GREEN":
        return [
            "Keep public calibration locked until an explicit operator-approved public run is planned.",
            "Use weakest private family rates to choose the next broader transfer ladder rather than adding benchmark-specific adapters.",
        ]
    if not current_evidence.get("fresh"):
        return [
            "Regenerate broad-private unattended candidate fanout, score, and gate under the current decoder source/release binary.",
            "Keep public calibration locked; stale private transfer evidence cannot support readiness.",
        ]
    if precise_blocker:
        kind = precise_blocker.get("kind")
        if kind == "candidate_coverage":
            return ["Repair broad private candidate coverage first; no-admissible failures block meaningful training score interpretation."]
        if kind == "broad_private_transfer_floor":
            return ["Cluster weakest private families and patch reusable decoder/learner paths before any public calibration."]
        if kind == "sts_causal_control":
            return ["Repair STS same-seed causal path before treating STS-default-on as promotion evidence."]
        if kind == "candidate_generation_missing":
            return ["Fix fanout/checkpoint prerequisites so private heldout candidate rows are produced."]
    if unattended.get("trigger_state") == "RED":
        return ["Fix unattended runner hard blocker, then rerun with --execute."]
    if blockers:
        return [f"Clear gate blocker `{blockers[0].get('gate')}` first."]
    return ["Run broad_private_generalization_unattended_v1.py --execute to produce score evidence."]


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def public_lock_evidence(path: Path) -> dict[str, Any]:
    reason = ""
    if path.exists():
        try:
            reason = path.read_text(encoding="utf-8").strip()
        except OSError:
            reason = "lock exists but could not be read"
    return {"active": path.exists(), "path": rel(path), "reason": reason}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Broad Private Generalization Gate V1",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Completion evidence: `{summary.get('completion_evidence_status')}`",
        f"- Heldout pass rate: {summary.get('heldout_pass_rate')}",
        f"- Heldout passes: {summary.get('heldout_passes')}/{summary.get('heldout_task_count')}",
        f"- No-admissible rate: {summary.get('no_admissible_task_rate')}",
        f"- STS delta: {summary.get('sts_delta')}",
        f"- STS regressions: {summary.get('sts_regressions')}",
        f"- Decoder source/release fresh: {summary.get('decoder_source_release_fresh')}",
        f"- Blockers: {summary.get('blocker_count')}",
        "",
        "## Next Actions",
    ]
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
