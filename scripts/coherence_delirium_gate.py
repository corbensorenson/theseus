"""Turn BugBrain's coherence/delirium health signal into active governance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_SOURCE = REPORTS / "coherence_delirium_report.json"
DEFAULT_OUT = REPORTS / "coherence_delirium_gate.json"
DEFAULT_MAX_DELIRIUM = 0.35
DEFAULT_CANDIDATE_MAX_DELIRIUM = 0.18


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--max-delirium", type=float, default=DEFAULT_MAX_DELIRIUM)
    parser.add_argument("--candidate-max-delirium", type=float, default=DEFAULT_CANDIDATE_MAX_DELIRIUM)
    args = parser.parse_args()

    source_path = resolve(args.source)
    gate = load_gate(
        source_path=source_path,
        max_delirium=args.max_delirium,
        candidate_max_delirium=args.candidate_max_delirium,
    )
    write_json(resolve(args.out), gate)
    print(json.dumps(gate, indent=2))
    return 0 if gate["allows_long_autonomy"] else 2


def load_gate(
    source_path: Path | None = None,
    *,
    max_delirium: float = DEFAULT_MAX_DELIRIUM,
    candidate_max_delirium: float = DEFAULT_CANDIDATE_MAX_DELIRIUM,
) -> dict[str, Any]:
    source_path = source_path or DEFAULT_SOURCE
    return evaluate_report(
        read_json(source_path),
        source_path=source_path,
        max_delirium=max_delirium,
        candidate_max_delirium=candidate_max_delirium,
    )


def evaluate_report(
    report: dict[str, Any],
    *,
    source_path: Path = DEFAULT_SOURCE,
    max_delirium: float = DEFAULT_MAX_DELIRIUM,
    candidate_max_delirium: float = DEFAULT_CANDIDATE_MAX_DELIRIUM,
) -> dict[str, Any]:
    trigger_state = str(report.get("trigger_state") or "MISSING")
    delirium_score = float_or(report.get("delirium_score"), default=1.0)
    coherence_score = float_or(report.get("coherence_score"), default=0.0)
    external_inference_calls = int_or(report.get("external_inference_calls"), default=0)
    penalties = [row for row in report.get("penalties", []) if isinstance(row, dict)]

    gates = [
        gate("coherence_report_available", bool(report), rel(source_path)),
        gate(
            "coherence_report_policy_known",
            report.get("policy") == "bugbrain_coherence_delirium_metric_v0",
            report.get("policy"),
        ),
        gate("coherence_not_red", trigger_state != "RED", trigger_state),
        gate("delirium_below_long_autonomy_threshold", delirium_score <= max_delirium, f"{delirium_score} <= {max_delirium}"),
        gate("external_inference_zero", external_inference_calls == 0, external_inference_calls),
    ]
    promotion_gates = [
        gate("coherence_green_for_candidate_promotion", trigger_state == "GREEN", trigger_state),
        gate(
            "delirium_below_candidate_threshold",
            delirium_score <= candidate_max_delirium,
            f"{delirium_score} <= {candidate_max_delirium}",
        ),
    ]
    failed = [row for row in gates if not row["passed"]]
    promotion_failed = failed + [row for row in promotion_gates if not row["passed"]]
    warnings = [
        f"{row.get('name')}={row.get('severity')}"
        for row in penalties
        if float_or(row.get("severity"), default=0.0) > 0.0
    ]
    allows_long_autonomy = not failed
    allows_candidate_promotion = not promotion_failed
    allows_self_edit = allows_long_autonomy
    allows_capability_expansion = allows_candidate_promotion
    trigger = "GREEN"
    if trigger_state == "YELLOW" and allows_long_autonomy:
        trigger = "YELLOW"
    if warnings and trigger_state != "GREEN":
        trigger = "YELLOW"
    if failed:
        trigger = "RED"

    return {
        "policy": "bugbrain_coherence_delirium_gate_v0",
        "created_utc": now(),
        "source_report": rel(source_path),
        "trigger_state": trigger,
        "source_trigger_state": trigger_state,
        "coherence_score": round(coherence_score, 6),
        "delirium_score": round(delirium_score, 6),
        "max_delirium": max_delirium,
        "candidate_max_delirium": candidate_max_delirium,
        "allows_long_autonomy": allows_long_autonomy,
        "allows_candidate_promotion": allows_candidate_promotion,
        "allows_self_edit": allows_self_edit,
        "allows_capability_expansion": allows_capability_expansion,
        "review_required": bool(failed or promotion_failed or trigger_state != "GREEN"),
        "blockers": [row["gate"] for row in failed],
        "candidate_blockers": [row["gate"] for row in promotion_failed],
        "warnings": warnings,
        "penalties": penalties,
        "gates": gates + promotion_gates,
        "runtime_contract": {
            "long_autonomy": "High delirium blocks long autonomous profile execution.",
            "candidate_promotion": "Candidate promotion requires a green coherence state and low delirium.",
            "self_edit": "Teacher/self-evolution apply mode is blocked by high delirium.",
            "capability_expansion": "Risky capability growth waits for clean coherence evidence.",
        },
        "next_actions": next_actions(failed, promotion_failed, warnings),
        "external_inference_calls": external_inference_calls,
    }


def next_actions(
    failed: list[dict[str, Any]],
    promotion_failed: list[dict[str, Any]],
    warnings: list[str],
) -> list[str]:
    if failed:
        return [f"Pause long autonomy and repair {row['gate']}: {row['evidence']}" for row in failed[:5]]
    candidate_only = [row for row in promotion_failed if row not in failed]
    if candidate_only:
        return [f"Keep training bounded; do not promote candidates until {row['gate']} clears." for row in candidate_only[:3]]
    if warnings:
        return ["Continue bounded work, but keep the coherence penalties visible in teacher and launch evidence."]
    return ["Coherence/delirium gate is clean for bounded autonomy."]


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_or(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
