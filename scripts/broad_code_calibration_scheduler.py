"""Pick the next broad public-code calibration target.

The broad transfer matrix is truth, but it is passive by itself. This scheduler
turns matrix blockers into the next concrete pressure card while keeping public
benchmarks calibration-only and preserving the no-cheat line.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import real_code_benchmark_graduation as real_code
import high_transfer_curriculum_scheduler as high_transfer
import high_transfer_scheduler_code_state as high_transfer_code_state


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_CALIBRATION_OPERATOR_LOCK = REPORTS / "public_calibration_operator_lock.flag"
PUBLIC_CALIBRATION_READINESS_PACKET = REPORTS / "public_calibration_readiness_packet.json"
DEFAULT_ORDER = [
    "source_mbpp",
    "source_evalplus",
    "source_human_eval",
    "source_bigcodebench",
    "source_livecodebench",
]
PUBLIC_TASK_CARDS = {
    "source_human_eval",
    "source_mbpp",
    "source_evalplus",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="reports/broad_transfer_matrix.json")
    parser.add_argument("--curriculum", default="reports/benchmaxx_curriculum.json")
    parser.add_argument("--min-public-tasks", type=int, default=32)
    parser.add_argument("--out", default="reports/broad_code_calibration_scheduler.json")
    parser.add_argument("--markdown-out", default="reports/broad_code_calibration_scheduler.md")
    args = parser.parse_args()

    matrix = read_json(resolve(args.matrix))
    curriculum = read_json(resolve(args.curriculum))
    ready_order = ready_code_order(curriculum) or DEFAULT_ORDER
    stalled_current = stalled_current_card(curriculum)
    selected = select_target(
        matrix,
        ready_order,
        min_public_tasks=max(1, args.min_public_tasks),
        skip_card=stalled_current,
    )
    private_receiver_gate = public_receiver_gate_state()
    public_calibration_operator_lock = public_calibration_operator_lock_state()
    if selected.get("can_run_real_code") and not private_receiver_gate["allowed"]:
        selected = block_for_private_receiver_gate(selected, private_receiver_gate)
    payload = {
        "policy": "project_theseus_broad_code_calibration_scheduler_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if selected["action"] == "no_action" else "YELLOW",
        "matrix": display(resolve(args.matrix)),
        "curriculum": display(resolve(args.curriculum)),
        "purpose": "Convert broad public-transfer blockers into the next honest code pressure target.",
        "selection_order": ready_order,
        "stalled_current_card_skipped": stalled_current,
        "selected": selected,
        "private_receiver_gate": private_receiver_gate,
        "public_calibration_operator_lock": public_calibration_operator_lock,
        "broad_matrix_summary": matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {},
        "rules": {
            "public_benchmarks": "calibration_only_not_training",
            "teacher_role": "architecture_diagnosis_only_no_answers",
            "minimum_public_task_slice": max(1, args.min_public_tasks),
            "loader_only_cards": "adapter_upgrade_or_loader_regression_only_not_promotion",
            "receiver_calibration_gate": "public calibration requires fresh private closure plus decoder_v2 private ablation gate; duplicate surfaces are controlled by the run registry",
            "legacy_operator_lock": "reported for audit only; fresh governed measurements are not calendar- or lock-throttled",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0


def public_receiver_gate_state() -> dict[str, Any]:
    """Mirror the Hive work-board gate so legacy schedulers cannot skip it."""

    readiness = read_json(PUBLIC_CALIBRATION_READINESS_PACKET)
    readiness_ready = bool(
        readiness.get("policy") == "project_theseus_public_calibration_readiness_packet_v1"
        and readiness.get("trigger_state") == "GREEN"
        and readiness.get("technical_ready_for_one_bounded_4_card_calibration") is True
    )
    closure = private_pressure_private_closure_state()
    ablation = decoder_v2_private_ablation_gate_state()
    allowed = bool(
        readiness_ready
        or (
            closure.get("allows_public_recalibration")
            and ablation.get("ready_for_public_calibration")
        )
    )
    blockers: list[str] = []
    if not allowed and not closure.get("allows_public_recalibration"):
        blockers.append(f"private_pressure_private_closure:{closure.get('reason')}")
    if not allowed and not ablation.get("ready_for_public_calibration"):
        blockers.append(f"decoder_v2_private_ablation_gate:{ablation.get('reason')}")
    if not allowed and not readiness_ready:
        blockers.append("public_calibration_readiness_packet:not_green_or_not_technical_ready")
    return {
        "allowed": allowed,
        "blockers": blockers,
        "readiness_packet": {
            "path": display(PUBLIC_CALIBRATION_READINESS_PACKET),
            "exists": PUBLIC_CALIBRATION_READINESS_PACKET.exists(),
            "trigger_state": readiness.get("trigger_state"),
            "technical_ready_for_one_bounded_4_card_calibration": readiness.get(
                "technical_ready_for_one_bounded_4_card_calibration"
            ),
            "operator_lock_active": readiness.get("operator_lock_active"),
            "summary": readiness.get("summary") if isinstance(readiness.get("summary"), dict) else {},
        },
        "private_pressure_private_closure": closure,
        "decoder_v2_private_ablation_gate": ablation,
        "score_semantics": "gate controls whether a public receiver calibration may be scheduled; public benchmarks remain calibration-only",
    }


def block_for_private_receiver_gate(selected: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    blocked = dict(selected)
    blocked["action"] = "private_gate_required_before_public_calibration"
    blocked["reason"] = "Public receiver calibration is blocked until fresh private closure and decoder_v2 private ablation gates are current."
    blocked["can_run_real_code"] = False
    blocked["teacher_due"] = False
    blocked["gate_blockers"] = list(gate.get("blockers") or [])
    blocked["blocked_selected_action"] = selected.get("action")
    return blocked


def private_pressure_private_closure_state() -> dict[str, Any]:
    helper = getattr(high_transfer, "private_pressure_private_closure_state", None)
    if callable(helper):
        return helper()
    helper = getattr(high_transfer_code_state, "private_pressure_private_closure_state", None)
    if callable(helper):
        return helper()
    return {
        "allows_public_recalibration": False,
        "reason": "private_pressure_private_closure_state_helper_missing",
    }


def decoder_v2_private_ablation_gate_state() -> dict[str, Any]:
    helper = getattr(high_transfer, "decoder_v2_private_ablation_gate_state", None)
    if callable(helper):
        return helper()
    helper = getattr(high_transfer_code_state, "decoder_v2_private_ablation_gate_state", None)
    if callable(helper):
        return helper()
    gate = read_json(REPORTS / "decoder_v2_private_ablation_gate.json")
    return {
        "ready_for_public_calibration": bool(gate.get("ready_for_public_calibration")),
        "reason": "direct_decoder_v2_private_ablation_gate_fallback",
        "ablation_report": display(REPORTS / "decoder_v2_private_ablation_gate.json"),
        "ablation_trigger_state": gate.get("trigger_state"),
    }


def select_target(
    matrix: dict[str, Any],
    ready_order: list[str],
    *,
    min_public_tasks: int,
    skip_card: str = "",
) -> dict[str, Any]:
    if matrix.get("policy") != "project_theseus_broad_transfer_matrix_v1":
        return target(
            "",
            "refresh_broad_transfer_matrix",
            "Broad transfer matrix is missing or not in the expected schema.",
            can_run_real_code=False,
            case_budget=min_public_tasks,
        )
    if matrix.get("trigger_state") == "GREEN":
        return target(
            "",
            "no_action",
            "Broad public-code transfer matrix is green.",
            can_run_real_code=False,
            case_budget=min_public_tasks,
        )

    rows = [row for row in matrix.get("rows", []) if isinstance(row, dict)]
    by_id = {str(row.get("card_id") or ""): row for row in rows}
    summary = matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {}
    below_floor = set(str(item) for item in summary.get("cards_below_floor", []) if str(item))
    no_clean = set(str(item) for item in summary.get("no_clean_student_evidence_cards", []) if str(item))
    loader_only = set(str(item) for item in summary.get("loader_only_cards", []) if str(item))
    coverage_warning = set(str(item) for item in summary.get("coverage_warning_cards", []) if str(item))

    ordered = [card_id for card_id in ready_order if card_id != skip_card]

    for card_id in ordered:
        row = by_id.get(card_id, {})
        if card_id in below_floor and card_id in PUBLIC_TASK_CARDS and int(row.get("public_task_count") or 0) >= min_public_tasks:
            capacity = public_task_capacity(card_id, min_public_tasks)
            return target(
                card_id,
                "run_public_calibration",
                "Card has clean student evidence but remains below the public transfer floor.",
                row=row,
                case_budget=min_public_tasks,
                source_capacity=capacity,
            )

    for card_id in ordered:
        row = by_id.get(card_id, {})
        if card_id in no_clean and card_id in PUBLIC_TASK_CARDS and int(row.get("public_task_count") or 0) >= min_public_tasks:
            capacity = public_task_capacity(card_id, min_public_tasks)
            return target(
                card_id,
                "run_public_calibration",
                "Card has sufficient public cases but no clean token-level student evidence.",
                row=row,
                case_budget=min_public_tasks,
                source_capacity=capacity,
            )

    for card_id in ordered:
        row = by_id.get(card_id, {})
        if card_id in no_clean and card_id in PUBLIC_TASK_CARDS and int(row.get("public_task_count") or 0) > 0:
            capacity = public_task_capacity(card_id, min_public_tasks)
            if int(row.get("public_task_count") or 0) < min_public_tasks and capacity["public_task_capacity"] >= min_public_tasks:
                return target(
                    card_id,
                    "run_public_calibration_expand_to_min",
                    "Previous report used a small slice, but the local source can supply the broad minimum; rerun at the minimum slice before judging transfer.",
                    row=row,
                    case_budget=min_public_tasks,
                    source_capacity=capacity,
                )
            if int(row.get("public_task_count") or 0) < min_public_tasks:
                return target(
                    card_id,
                    "stage_or_upgrade_public_task_adapter",
                    "Card has some public cases but cannot yet supply the broad minimum; stage/upgrade the adapter before spending learning cycles.",
                    row=row,
                    can_run_real_code=False,
                    case_budget=min_public_tasks,
                    source_capacity=capacity,
                )
            return target(
                card_id,
                "run_public_calibration_coverage_limited",
                "Card has public cases but the local slice is below the broad minimum; run it as calibration and keep the coverage warning.",
                row=row,
                case_budget=min_public_tasks,
            )

    for card_id in ordered:
        row = by_id.get(card_id, {})
        if card_id in coverage_warning and card_id in PUBLIC_TASK_CARDS:
            capacity = public_task_capacity(card_id, min_public_tasks)
            if capacity["public_task_capacity"] < min_public_tasks:
                return target(
                    card_id,
                    "stage_or_upgrade_public_task_adapter",
                    "Coverage warning points to a source/adapter capacity wall; fix that before rerunning calibration.",
                    row=row,
                    can_run_real_code=False,
                    case_budget=min_public_tasks,
                    source_capacity=capacity,
                )
            return target(
                card_id,
                "run_public_calibration",
                "Card has a coverage warning; rerun at the broad minimum if local source supports it.",
                row=row,
                case_budget=min_public_tasks,
                source_capacity=capacity,
            )

    for card_id in ordered:
        row = by_id.get(card_id, {})
        if card_id in loader_only:
            return target(
                card_id,
                "upgrade_public_task_adapter",
                "Card is loader-only, so it cannot be promotion evidence until a real public-task adapter exists.",
                row=row,
                can_run_real_code=False,
                case_budget=min_public_tasks,
            )

    if skip_card and skip_card in by_id:
        return target(
            skip_card,
            "defer_stalled_current_no_better_broad_target",
            "The current broad wall is stalled, but no alternate ready public-code blocker was actionable.",
            row=by_id.get(skip_card, {}),
            can_run_real_code=False,
            case_budget=min_public_tasks,
        )

    return target(
        "",
        "review_broad_matrix",
        "Broad matrix is yellow but no ready card matched an automatic calibration action.",
        can_run_real_code=False,
        case_budget=min_public_tasks,
    )


def target(
    card_id: str,
    action: str,
    reason: str,
    *,
    row: dict[str, Any] | None = None,
    can_run_real_code: bool | None = None,
    case_budget: int = 32,
    source_capacity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = row or {}
    real_code_allowed = bool(card_id in PUBLIC_TASK_CARDS and action.startswith("run_public_calibration"))
    if can_run_real_code is not None:
        real_code_allowed = bool(can_run_real_code)
    return {
        "card_id": card_id,
        "action": action,
        "reason": reason,
        "runner_family": "coding_local_sandbox" if card_id else "",
        "case_budget": max(1, int(case_budget)),
        "can_run_real_code": real_code_allowed,
        "public_calibration_only": bool(card_id),
        "teacher_due": action in {"upgrade_public_task_adapter", "stage_or_upgrade_public_task_adapter", "review_broad_matrix"},
        "status": row.get("status"),
        "public_task_count": int(row.get("public_task_count") or 0),
        "source_public_task_capacity": (source_capacity or {}).get("public_task_capacity"),
        "source_evidence_level": (source_capacity or {}).get("benchmark_evidence_level"),
        "source_capacity_detail": source_capacity or {},
        "multi_stream_pass_rate": float(row.get("multi_stream_pass_rate") or 0.0),
        "single_stream_pass_rate": float(row.get("single_stream_pass_rate") or 0.0),
        "coverage_warnings": row.get("coverage_warnings") or [],
        "no_cheat_violations": row.get("no_cheat_violations") or [],
        "clean_evidence_blockers": row.get("clean_evidence_blockers") or [],
        "selected_report": row.get("selected_report") or "",
    }


def ready_code_order(curriculum: dict[str, Any]) -> list[str]:
    next_frontier = curriculum.get("next_frontier") if isinstance(curriculum.get("next_frontier"), dict) else {}
    rotation = next_frontier.get("same_family_rotation") if isinstance(next_frontier.get("same_family_rotation"), dict) else {}
    ready = [str(item) for item in rotation.get("ready_order", []) if str(item)]
    ordered = [card for card in DEFAULT_ORDER if card in ready]
    ordered.extend(card for card in ready if card not in ordered)
    return ordered


def public_task_capacity(card_id: str, min_public_tasks: int) -> dict[str, Any]:
    card = read_json(ROOT / "benchmarks" / "cards" / f"{card_id}.json")
    source_id = str(card.get("source_id") or card_id.replace("source_", ""))
    source_path = real_code.resolve_source_path(card)
    tasks, evidence_level, semantics = real_code.load_cases(
        card_id,
        source_id,
        source_path,
        seed=14,
        max_cases=max(1, int(min_public_tasks)),
    )
    public_task_count = len(tasks) if evidence_level == "public_benchmark_task_regression" else 0
    return {
        "card_id": card_id,
        "source_id": source_id,
        "source_path": str(source_path).replace("\\", "/"),
        "benchmark_evidence_level": evidence_level,
        "score_semantics": semantics,
        "public_task_capacity": public_task_count,
        "meets_minimum_public_slice": public_task_count >= max(1, int(min_public_tasks)),
    }


def stalled_current_card(curriculum: dict[str, Any]) -> str:
    next_frontier = curriculum.get("next_frontier") if isinstance(curriculum.get("next_frontier"), dict) else {}
    rotation = next_frontier.get("same_family_rotation") if isinstance(next_frontier.get("same_family_rotation"), dict) else {}
    transfer = rotation.get("public_code_transfer_stall") if isinstance(rotation.get("public_code_transfer_stall"), dict) else {}
    if bool(transfer.get("stalled")):
        return str(rotation.get("current_card_id") or rotation.get("selected_card_id") or "")
    return ""


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def public_calibration_operator_lock_state() -> dict[str, Any]:
    active = PUBLIC_CALIBRATION_OPERATOR_LOCK.exists()
    reason = ""
    if active:
        try:
            reason = PUBLIC_CALIBRATION_OPERATOR_LOCK.read_text(encoding="utf-8").strip()
        except OSError:
            reason = "operator lock file exists but could not be read"
    return {
        "active": active,
        "path": display(PUBLIC_CALIBRATION_OPERATOR_LOCK),
        "reason": reason,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    selected = payload.get("selected") if isinstance(payload.get("selected"), dict) else {}
    summary = payload.get("broad_matrix_summary") if isinstance(payload.get("broad_matrix_summary"), dict) else {}
    private_gate = payload.get("private_receiver_gate") if isinstance(payload.get("private_receiver_gate"), dict) else {}
    lines = [
        "# Broad Code Calibration Scheduler",
        "",
        f"Generated: {payload.get('created_utc')}",
        f"Trigger: **{payload.get('trigger_state')}**",
        "",
        f"- Selected card: {selected.get('card_id') or 'none'}",
        f"- Action: {selected.get('action')}",
        f"- Reason: {selected.get('reason')}",
        f"- Case budget: {selected.get('case_budget')}",
        f"- Can run real-code calibration: {selected.get('can_run_real_code')}",
        f"- Private receiver gate: {private_gate.get('allowed')}",
        f"- Gate blockers: {', '.join(private_gate.get('blockers') or []) or 'none'}",
        "",
        f"- Broad public pass rate: {summary.get('real_public_pass_rate')}",
        f"- Below-floor cards: {', '.join(summary.get('cards_below_floor') or []) or 'none'}",
        f"- No clean evidence: {', '.join(summary.get('no_clean_student_evidence_cards') or []) or 'none'}",
        f"- Loader-only cards: {', '.join(summary.get('loader_only_cards') or []) or 'none'}",
        "",
        "Public benchmarks remain calibration-only. This scheduler never turns public answers into training data.",
        "",
    ]
    return "\n".join(lines)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
