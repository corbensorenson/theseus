"""Private-pressure promotion gate for multi-stream experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pressure-report", default="")
    parser.add_argument("--monitorability-report", default="reports/multi_stream_monitorability_probe.json")
    parser.add_argument("--real-regression-report", default="")
    parser.add_argument("--floor", type=float, default=0.70)
    parser.add_argument("--out", default="reports/multi_stream_candidate_gate.json")
    args = parser.parse_args()

    pressure_path = resolve(args.pressure_report) if args.pressure_report else latest(REPORTS, "multi_stream_code_pressure_*_seed*.json")
    pressure = read_json(pressure_path)
    monitorability = read_json(resolve(args.monitorability_report))
    real_regression = read_json(resolve(args.real_regression_report)) if args.real_regression_report else {}
    score = as_float(pressure.get("score"), 0.0)
    pass_rate = as_float(get_path(pressure, ["summary", "multi_stream_pass_rate"], 0.0), 0.0)
    pass_rate_delta = as_float(get_path(pressure, ["summary", "pass_rate_delta"], 0.0), 0.0)
    task_improvements = int(as_float(get_path(pressure, ["summary", "task_level_improvements_over_single_stream"], 0), 0.0))
    task_regressions = int(as_float(get_path(pressure, ["summary", "task_level_regressions_vs_single_stream"], 0), 0.0))
    patch_synthesis_used = int(as_float(get_path(pressure, ["summary", "patch_stream_synthesis_used_count"], 0), 0.0))
    semantics = str(
        pressure.get("score_semantics")
        or get_path(pressure, ["summary", "score_semantics"], "")
        or get_path(pressure, ["metrics", "score_semantics"], "")
        or "private_multistream_pressure_correctness_monitorability_and_critical_path"
    )
    benchmark_evidence_level = str(
        pressure.get("benchmark_evidence_level")
        or get_path(pressure, ["summary", "benchmark_evidence_level"], "")
        or ""
    )
    public_score_claim = str(
        pressure.get("public_benchmark_score_claim")
        or get_path(pressure, ["summary", "public_benchmark_score_claim"], "")
        or ""
    )
    private_pressure_detected = "private_multistream_pressure" in semantics or str(pressure.get("card_id") or "").startswith("multistream_")
    public_loader_regression_detected = (
        "public_loader_regression" in semantics
        or benchmark_evidence_level == "public_loader_regression"
        or public_score_claim == "forbidden"
    )
    non_promotable_pressure_detected = private_pressure_detected or public_loader_regression_detected
    real_regression_holds = bool(real_regression) and as_float(get_path(real_regression, ["summary", "accuracy"], real_regression.get("score")), 0.0) >= args.floor
    promotion_allowed = bool(
        pressure.get("policy") == "project_theseus_multi_stream_code_pressure_v1"
        and score >= args.floor
        and pass_rate >= args.floor
        and not non_promotable_pressure_detected
        and real_regression_holds
    )
    correctly_blocked = non_promotable_pressure_detected and not promotion_allowed
    gates = [
        gate("multi_stream_pressure_present", pressure.get("policy") == "project_theseus_multi_stream_code_pressure_v1", rel(pressure_path)),
        gate("causal_verifier_green", get_path(pressure, ["verifier", "trigger_state"], "") == "GREEN", get_path(pressure, ["verifier", "trigger_state"], None)),
        gate("monitorability_probe_green", monitorability.get("trigger_state") == "GREEN", monitorability.get("trigger_state")),
        gate(
            "patch_selection_improves_over_single_stream",
            pass_rate_delta > 0.0 and task_improvements > 0 and task_regressions == 0,
            f"delta={pass_rate_delta:.6f} improvements={task_improvements} regressions={task_regressions}",
        ),
        gate("patch_stream_synthesis_observed", patch_synthesis_used > 0, f"synthesis_used={patch_synthesis_used}"),
        gate(
            "non_promotable_pressure_detected",
            non_promotable_pressure_detected,
            f"semantics={semantics} evidence={benchmark_evidence_level} public_score_claim={public_score_claim}",
        ),
        gate(
            "promotion_blocked_without_real_regression",
            correctly_blocked,
            "multi-stream pressure can train/rotate but cannot promote without true benchmark-regression evidence",
        ),
        gate("external_inference_zero", int(pressure.get("external_inference_calls") or 0) == 0, pressure.get("external_inference_calls")),
    ]
    trigger_state = "GREEN" if all(item["passed"] for item in gates) else "RED"
    report = {
        "policy": "project_theseus_multi_stream_candidate_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "pressure_report": rel(pressure_path),
        "monitorability_report": rel(resolve(args.monitorability_report)),
        "decision": {
            "promotion_allowed": promotion_allowed,
            "correctly_blocked_private_pressure": correctly_blocked,
            "score": score,
            "floor": args.floor,
            "multi_stream_pass_rate": pass_rate,
            "pass_rate_delta": pass_rate_delta,
            "task_level_improvements_over_single_stream": task_improvements,
            "task_level_regressions_vs_single_stream": task_regressions,
            "patch_stream_synthesis_used_count": patch_synthesis_used,
            "private_pressure_detected": private_pressure_detected,
            "public_loader_regression_detected": public_loader_regression_detected,
            "non_promotable_pressure_detected": non_promotable_pressure_detected,
            "score_semantics": semantics,
            "benchmark_evidence_level": benchmark_evidence_level,
            "public_benchmark_score_claim": public_score_claim,
            "real_regression_holds": real_regression_holds,
            "rule": "multi_stream_pressure_can_train_and_rotate_but_cannot_promote_without_real_benchmark_regression",
        },
        "gates": gates,
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if trigger_state == "GREEN" else 1


def latest(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else directory / "__missing__.json"


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


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


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
