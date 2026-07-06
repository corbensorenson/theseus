#!/usr/bin/env python3
"""Compare VCM-on and VCM-off broad survival-lane runs.

VCM should be an integral Theseus context substrate, but a consumer must still
prove it helps or is neutral under equal budget. This report gates VCM off for
the current body-template selector when evidence shows harm, while keeping the
feature path available for future structural/full-body consumers.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ON = ROOT / "reports" / "broad_capability_survival_lane_v1_vcm_on_smoke.json"
DEFAULT_OFF = ROOT / "reports" / "broad_capability_survival_lane_v1_vcm_off_smoke.json"
DEFAULT_OUT = ROOT / "reports" / "broad_capability_vcm_feature_ablation_v1.json"
DEFAULT_MD = ROOT / "reports" / "broad_capability_vcm_feature_ablation_v1.md"
DEFAULT_POLICY = ROOT / "configs" / "vcm_broad_survival_feature_policy_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcm-on", default=rel(DEFAULT_ON))
    parser.add_argument("--vcm-off", default=rel(DEFAULT_OFF))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--policy-out", default=rel(DEFAULT_POLICY))
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_json(resolve(args.policy_out), report["recommended_policy"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    on = read_json(resolve(args.vcm_on))
    off = read_json(resolve(args.vcm_off))
    on_summary = run_summary(on)
    off_summary = run_summary(off)
    deltas = {
        "transformer_control": round(on_summary["transformer"] - off_summary["transformer"], 6),
        "symliquid_style": round(on_summary["symliquid"] - off_summary["symliquid"], 6),
    }
    harmed = [arm for arm, delta in deltas.items() if delta < 0.0]
    on_clean = gates_clean(on)
    off_clean = gates_clean(off)
    policy = recommended_policy(args, on, off, deltas, harmed)
    gates = [
        gate("vcm_on_report_loaded", bool(on), rel(resolve(args.vcm_on)), "hard"),
        gate("vcm_off_report_loaded", bool(off), rel(resolve(args.vcm_off)), "hard"),
        gate("vcm_on_run_clean", on_clean, failed_gates(on), "hard"),
        gate("vcm_off_run_clean", off_clean, failed_gates(off), "hard"),
        gate("vcm_on_consumed_context", bool(get_path(on, ["summary", "vcm_context_active"], False)), get_path(on, ["summary", "vcm_rows_with_context"], 0), "hard"),
        gate("vcm_off_no_context", not bool(get_path(off, ["summary", "vcm_context_active"], False)), get_path(off, ["summary", "vcm_rows_with_context"], 0), "hard"),
        gate("public_training_rows_zero", public_training_rows(on, off) == 0, public_training_rows(on, off), "hard"),
        gate("external_inference_zero", external_calls(on, off) == 0, external_calls(on, off), "hard"),
        gate("fallback_return_zero", fallback_returns(on, off) == 0, fallback_returns(on, off), "hard"),
        gate("vcm_neutral_or_gated", not harmed or policy["action"].startswith("disable_"), {"deltas": deltas, "harmed": harmed, "policy": policy["action"]}, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    trigger_state = "GREEN" if not hard_failed and not harmed else ("YELLOW" if not hard_failed else "RED")
    return {
        "policy": "project_theseus_broad_capability_vcm_feature_ablation_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "vcm_on": rel(resolve(args.vcm_on)),
            "vcm_off": rel(resolve(args.vcm_off)),
        },
        "summary": {
            "vcm_on_transformer_pass_rate": on_summary["transformer"],
            "vcm_off_transformer_pass_rate": off_summary["transformer"],
            "vcm_on_symliquid_pass_rate": on_summary["symliquid"],
            "vcm_off_symliquid_pass_rate": off_summary["symliquid"],
            "deltas": deltas,
            "harmed_arms": harmed,
            "vcm_rows_with_context": get_path(on, ["summary", "vcm_rows_with_context"], 0),
            "vcm_unique_context_hashes": get_path(on, ["summary", "vcm_unique_context_hashes"], []),
            "recommended_action": policy["action"],
            "policy_out": rel(resolve(args.policy_out)),
            "external_inference_calls": 0,
            "public_training_rows_written": 0,
            "fallback_return_count": 0,
        },
        "recommended_policy": policy,
        "gates": gates,
        "score_semantics": (
            "Private equal-budget VCM feature ablation for the broad survival lane. This is not public "
            "calibration or promotion evidence. Harm gates VCM off for the current body-template selector."
        ),
        "external_inference_calls": 0,
    }


def run_summary(report: dict[str, Any]) -> dict[str, float]:
    return {
        "transformer": float(get_path(report, ["summary", "transformer_sts_on_pass_rate"], 0.0) or 0.0),
        "symliquid": float(get_path(report, ["summary", "symliquid_sts_on_pass_rate"], 0.0) or 0.0),
    }


def recommended_policy(
    args: argparse.Namespace,
    on: dict[str, Any],
    off: dict[str, Any],
    deltas: dict[str, float],
    harmed: list[str],
) -> dict[str, Any]:
    disable = bool(harmed)
    return {
        "policy": "project_theseus_vcm_broad_survival_feature_policy_v1",
        "created_utc": now(),
        "action": "disable_vcm_for_broad_body_template_selector" if disable else "allow_vcm_for_broad_body_template_selector",
        "applies_to": [
            "project_theseus_broad_capability_survival_lane_comparator_v1",
            "private_train_body_template_selector",
        ],
        "default_vcm_mode": "off" if disable else "auto",
        "evidence": {
            "vcm_on": rel(resolve(args.vcm_on)),
            "vcm_off": rel(resolve(args.vcm_off)),
            "deltas": deltas,
            "harmed_arms": harmed,
            "vcm_rows_with_context": get_path(on, ["summary", "vcm_rows_with_context"], 0),
            "vcm_unique_context_hashes": get_path(on, ["summary", "vcm_unique_context_hashes"], []),
        },
        "re_enable_rule": (
            "VCM may be re-enabled by default for this comparator only after an equal-budget private "
            "ablation shows non-negative transformer_control and symliquid_style deltas. Structural/full-body "
            "generators may run VCM under explicit --vcm-mode on for their own ablations."
        ),
        "public_calibration_allowed": False,
        "external_inference_calls": 0,
    }


def gates_clean(report: dict[str, Any]) -> bool:
    return bool(report) and report.get("trigger_state") in {"GREEN", "YELLOW"} and not failed_gates(report)


def failed_gates(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in report.get("gates", []) if isinstance(row, dict) and not row.get("passed")]


def public_training_rows(*reports: dict[str, Any]) -> int:
    return sum(int(get_path(report, ["summary", "public_benchmark_training_rows"], 0) or 0) for report in reports)


def external_calls(*reports: dict[str, Any]) -> int:
    return sum(int(report.get("external_inference_calls") or get_path(report, ["summary", "external_inference_calls"], 0) or 0) for report in reports)


def fallback_returns(*reports: dict[str, Any]) -> int:
    return sum(int(get_path(report, ["summary", "fallback_return_count"], 0) or 0) for report in reports)


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cursor = value
    for part in path:
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Broad Capability VCM Feature Ablation v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- transformer VCM-on/off: `{summary.get('vcm_on_transformer_pass_rate')}` / `{summary.get('vcm_off_transformer_pass_rate')}`",
        f"- SymLiquid VCM-on/off: `{summary.get('vcm_on_symliquid_pass_rate')}` / `{summary.get('vcm_off_symliquid_pass_rate')}`",
        f"- deltas: `{summary.get('deltas')}`",
        f"- recommended_action: `{summary.get('recommended_action')}`",
        "",
        "## Failed Gates",
    ]
    failed = [row for row in report.get("gates", []) if isinstance(row, dict) and not row.get("passed")]
    if not failed:
        lines.append("- none")
    else:
        for row in failed:
            lines.append(f"- `{row.get('name')}` ({row.get('severity')}): `{row.get('evidence')}`")
    return "\n".join(lines) + "\n"


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
