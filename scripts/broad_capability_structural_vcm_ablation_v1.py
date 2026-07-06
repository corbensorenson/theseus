#!/usr/bin/env python3
"""Ablate VCM on the promotion-grade structural-action path.

This gate compares matched VCM-off and VCM-on structural-action reports. It
writes an explicit route policy so VCM is used only where it helps or is
neutral. Public calibration, public training rows, teacher calls, external
inference, fallback returns, and runtime serving are all out of scope.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VCM_OFF = ROOT / "reports" / "broad_capability_structural_action_decoder_probe_v1_vcm_off.json"
DEFAULT_VCM_ON = ROOT / "reports" / "broad_capability_structural_action_decoder_probe_v1_vcm_on.json"
DEFAULT_OUT = ROOT / "reports" / "broad_capability_structural_vcm_ablation_v1.json"
DEFAULT_MD = ROOT / "reports" / "broad_capability_structural_vcm_ablation_v1.md"
DEFAULT_POLICY = ROOT / "configs" / "vcm_structural_survival_feature_policy_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcm-off", default=rel(DEFAULT_VCM_OFF))
    parser.add_argument("--vcm-on", default=rel(DEFAULT_VCM_ON))
    parser.add_argument("--policy-out", default=rel(DEFAULT_POLICY))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_json(resolve(args.policy_out), report["route_policy"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    off_path = resolve(args.vcm_off)
    on_path = resolve(args.vcm_on)
    off = read_json(off_path)
    on = read_json(on_path)

    arms = {}
    for arm_id in ["transformer_control", "symliquid_style"]:
        arms[arm_id] = compare_arm(off, on, arm_id)

    transformer = arms["transformer_control"]
    symliquid = arms["symliquid_style"]
    transformer_ok = transformer["summary"]["structural_only_delta"] >= 0.0 and transformer["summary"]["augmented_delta"] >= 0.0
    symliquid_ok = symliquid["summary"]["structural_only_delta"] >= 0.0 and symliquid["summary"]["augmented_delta"] >= 0.0
    transformer_has_family_help = bool(transformer["family_gate"]["enabled_families"])

    if transformer_ok:
        action = "enable_vcm_for_promoted_transformer_structural_path"
        promoted_mode = "on"
        rationale = "VCM-on is neutral or better for transformer structural-only and augmented scores."
    elif transformer_has_family_help:
        action = "family_gate_vcm_for_promoted_transformer_structural_path"
        promoted_mode = "family_gated_default_off"
        rationale = "VCM-on harms transformer overall but helps at least one evaluated family; keep default off and allow explicit family overrides only."
    else:
        action = "disable_vcm_for_promoted_transformer_structural_path"
        promoted_mode = "off"
        rationale = "VCM-on harms the promoted transformer structural path and does not produce reliable family-level wins."

    route_policy = {
        "policy": "project_theseus_vcm_structural_survival_feature_policy_v1",
        "created_utc": now(),
        "action": action,
        "promoted_transformer_vcm_mode": promoted_mode,
        "symliquid_discovery_vcm_mode": "on" if symliquid_ok else "off",
        "feature_contract": "project_theseus_structural_action_vcm_feature_contract_v2",
        "model_visible_features": [
            "vcm_context.task_family_id",
            "vcm_context.selected_page_lanes",
            "vcm_context.retrieval_confidence_bucket",
            "vcm_context.task_family_memory_lane",
            "vcm_context.selected_page_count",
        ],
        "audit_only_features": [
            "vcm_context.selected_context_hash",
            "vcm_context.selected_page_titles",
            "vcm_context.selected_page_sources",
        ],
        "transformer_control": {
            "summary": transformer["summary"],
            "enabled_families": transformer["family_gate"]["enabled_families"],
            "disabled_families": transformer["family_gate"]["disabled_families"],
        },
        "symliquid_style": {
            "summary": symliquid["summary"],
        },
        "default_serving_allowed": False,
        "public_calibration_allowed": False,
        "teacher_used": False,
        "external_inference_calls": 0,
        "public_training_rows": 0,
        "fallback_return_count": 0,
        "rationale": rationale,
    }

    no_cheat = no_cheat_summary(off, on)
    gates = [
        gate("vcm_off_report_green", off.get("trigger_state") == "GREEN", off.get("trigger_state"), "hard"),
        gate("vcm_on_report_green", on.get("trigger_state") == "GREEN", on.get("trigger_state"), "hard"),
        gate("matched_eval_rows", get_path(off, ["summary", "eval_rows"]) == get_path(on, ["summary", "eval_rows"]) and int(get_path(on, ["summary", "eval_rows"], 0) or 0) >= 192, {"off": get_path(off, ["summary", "eval_rows"]), "on": get_path(on, ["summary", "eval_rows"])}, "hard"),
        gate("vcm_off_context_zero", int(get_path(off, ["summary", "vcm_rows_with_context"], 0) or 0) == 0, get_path(off, ["summary", "vcm_rows_with_context"]), "hard"),
        gate("vcm_on_context_consumed", int(get_path(on, ["summary", "vcm_rows_with_context"], 0) or 0) > 0, get_path(on, ["summary", "vcm_rows_with_context"]), "hard"),
        gate("policy_action_explicit", bool(action), action, "hard"),
        gate("no_public_training_rows", no_cheat["public_training_rows"] == 0, no_cheat["public_training_rows"], "hard"),
        gate("no_external_inference", no_cheat["external_inference_calls"] == 0, no_cheat["external_inference_calls"], "hard"),
        gate("no_teacher", no_cheat["teacher_used"] == 0, no_cheat["teacher_used"], "hard"),
        gate("no_fallback_returns", no_cheat["fallback_return_count"] == 0, no_cheat["fallback_return_count"], "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    trigger_state = "GREEN" if not hard_failed else "RED"
    if trigger_state == "GREEN" and action != "enable_vcm_for_promoted_transformer_structural_path":
        trigger_state = "YELLOW"

    return {
        "policy": "project_theseus_broad_capability_structural_vcm_ablation_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "recommended_action": action,
            "promoted_transformer_vcm_mode": promoted_mode,
            "symliquid_discovery_vcm_mode": route_policy["symliquid_discovery_vcm_mode"],
            "transformer_structural_only_delta": transformer["summary"]["structural_only_delta"],
            "transformer_augmented_delta": transformer["summary"]["augmented_delta"],
            "symliquid_structural_only_delta": symliquid["summary"]["structural_only_delta"],
            "symliquid_augmented_delta": symliquid["summary"]["augmented_delta"],
            "vcm_rows_with_context": get_path(on, ["summary", "vcm_rows_with_context"], 0),
            "unique_context_hashes": get_path(on, ["summary", "vcm_unique_context_hashes"], []),
            "harmed_arms": [arm for arm, row in arms.items() if row["summary"]["augmented_delta"] < 0.0 or row["summary"]["structural_only_delta"] < 0.0],
            "helped_arms": [arm for arm, row in arms.items() if row["summary"]["augmented_delta"] >= 0.0 and row["summary"]["structural_only_delta"] >= 0.0],
        },
        "arms": arms,
        "route_policy": route_policy,
        "no_cheat": no_cheat,
        "gates": gates,
        "inputs": {
            "vcm_off": rel(off_path),
            "vcm_on": rel(on_path),
            "policy_out": rel(resolve(args.policy_out)),
        },
        "score_semantics": (
            "Private structural-action VCM ablation. Public benchmark payloads and public calibration are not used. "
            "VCM hashes/pages are retained for audit; only stable semantic VCM fields may enter model-visible features."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def compare_arm(off: dict[str, Any], on: dict[str, Any], arm_id: str) -> dict[str, Any]:
    off_summary = dict_or_empty(get_path(off, ["arms", arm_id, "summary"], {}))
    on_summary = dict_or_empty(get_path(on, ["arms", arm_id, "summary"], {}))
    summary = {
        "baseline_off": number(off_summary.get("baseline_pass_rate")),
        "baseline_on": number(on_summary.get("baseline_pass_rate")),
        "structural_only_off": number(off_summary.get("structural_only_pass_rate")),
        "structural_only_on": number(on_summary.get("structural_only_pass_rate")),
        "augmented_off": number(off_summary.get("augmented_pass_rate")),
        "augmented_on": number(on_summary.get("augmented_pass_rate")),
        "structural_only_delta": delta(on_summary.get("structural_only_pass_rate"), off_summary.get("structural_only_pass_rate")),
        "augmented_delta": delta(on_summary.get("augmented_pass_rate"), off_summary.get("augmented_pass_rate")),
        "baseline_delta": delta(on_summary.get("baseline_pass_rate"), off_summary.get("baseline_pass_rate")),
        "fallback_return_rows_on": int(on_summary.get("fallback_return_rows") or 0),
        "syntax_pass_rate_on": number(on_summary.get("syntax_pass_rate")),
        "rank_pool_size": on_summary.get("rank_pool_size"),
        "compatibility_rerank": on_summary.get("compatibility_rerank"),
    }
    structural_family = family_delta(
        get_path(off, ["arms", arm_id, "private_verifier", "structural_only", "concept_family_pass_rates"], {}),
        get_path(on, ["arms", arm_id, "private_verifier", "structural_only", "concept_family_pass_rates"], {}),
    )
    augmented_family = family_delta(
        get_path(off, ["arms", arm_id, "private_verifier", "augmented", "concept_family_pass_rates"], {}),
        get_path(on, ["arms", arm_id, "private_verifier", "augmented", "concept_family_pass_rates"], {}),
    )
    enabled = [
        family
        for family, row in structural_family.items()
        if float(row.get("delta") or 0.0) > 0.0 and float(augmented_family.get(family, {}).get("delta") or 0.0) >= 0.0
    ]
    disabled = [
        family
        for family, row in structural_family.items()
        if float(row.get("delta") or 0.0) < 0.0 or float(augmented_family.get(family, {}).get("delta") or 0.0) < 0.0
    ]
    return {
        "summary": summary,
        "family_gate": {
            "enabled_families": sorted(enabled),
            "disabled_families": sorted(disabled),
            "structural_only_family_deltas": top_family_deltas(structural_family),
            "augmented_family_deltas": top_family_deltas(augmented_family),
        },
    }


def family_delta(off_rates: Any, on_rates: Any) -> dict[str, dict[str, float]]:
    off_map = dict_or_empty(off_rates)
    on_map = dict_or_empty(on_rates)
    out = {}
    for family in sorted(set(off_map) | set(on_map)):
        off_value = number(off_map.get(family))
        on_value = number(on_map.get(family))
        out[family] = {
            "off": off_value,
            "on": on_value,
            "delta": round((on_value or 0.0) - (off_value or 0.0), 6),
        }
    return out


def top_family_deltas(rows: dict[str, dict[str, float]]) -> dict[str, list[dict[str, Any]]]:
    values = [{"family": key, **value} for key, value in rows.items() if float(value.get("delta") or 0.0) != 0.0]
    values.sort(key=lambda row: float(row["delta"]))
    return {
        "most_negative": values[:12],
        "most_positive": list(reversed(values[-12:])),
    }


def no_cheat_summary(*reports: dict[str, Any]) -> dict[str, int]:
    text = json.dumps(reports, sort_keys=True).lower()
    return {
        "public_training_rows": sum_int_key(reports, "public_training_rows") + sum_int_key(reports, "public_training_rows_written"),
        "external_inference_calls": sum_int_key(reports, "external_inference_calls"),
        "teacher_used": 1 if '"teacher_used": true' in text else 0,
        "fallback_return_count": sum_int_key(reports, "fallback_return_count") + sum_int_key(reports, "fallback_return_rows"),
    }


def sum_int_key(values: Any, key: str) -> int:
    if isinstance(values, dict):
        total = 0
        for child_key, child in values.items():
            if child_key == key:
                total += int(number(child) or 0)
            else:
                total += sum_int_key(child, key)
        return total
    if isinstance(values, list) or isinstance(values, tuple):
        return sum(sum_int_key(child, key) for child in values)
    return 0


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence, "severity": severity}


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Broad Capability Structural VCM Ablation v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- recommended_action: `{summary.get('recommended_action')}`",
        f"- promoted_transformer_vcm_mode: `{summary.get('promoted_transformer_vcm_mode')}`",
        f"- symliquid_discovery_vcm_mode: `{summary.get('symliquid_discovery_vcm_mode')}`",
        f"- transformer_structural_only_delta: `{summary.get('transformer_structural_only_delta')}`",
        f"- transformer_augmented_delta: `{summary.get('transformer_augmented_delta')}`",
        f"- symliquid_structural_only_delta: `{summary.get('symliquid_structural_only_delta')}`",
        f"- symliquid_augmented_delta: `{summary.get('symliquid_augmented_delta')}`",
        "",
        "## Gates",
    ]
    for row in report.get("gates") or []:
        lines.append(f"- `{row.get('name')}`: `{row.get('passed')}` / `{row.get('evidence')}`")
    return "\n".join(lines) + "\n"


def get_path(obj: Any, parts: list[str], default: Any = None) -> Any:
    cur = obj
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def delta(on_value: Any, off_value: Any) -> float:
    return round((number(on_value) or 0.0) - (number(off_value) or 0.0), 6)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
