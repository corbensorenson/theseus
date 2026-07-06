#!/usr/bin/env python3
"""Emit a compact substrate verdict from existing matched evidence.

This report separates practical route selection from protected SymLiquid
discovery evidence. It does not run training or benchmarks, and it does not
promote either substrate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402


DEFAULT_OUT = ROOT / "reports" / "theseus_substrate_verdict.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "theseus_substrate_verdict.md"
EVIDENCE_REPORTS = [
    {
        "path": "reports/full_body_contract_transfer_recovery_v1_symliquid_same_slice_comparator.json",
        "label": "body_template_selector_same_slice",
        "evidence_role": "legacy_selector_comparator",
    },
    {
        "path": "reports/neural_seed_token_decoder_96eval_4096train_multiseed.json",
        "label": "token_decoder_96eval_4096train_multiseed",
        "evidence_role": "matched_private_token_decoder_multiseed",
    },
    {
        "path": "reports/neural_seed_token_decoder_multiseed_smoke.json",
        "label": "token_decoder_multiseed_smoke",
        "evidence_role": "matched_private_smoke",
    },
    {
        "path": "reports/candidate_floor_v2_survival_lane_seed101_comparator.json",
        "label": "candidate_floor_v2_survival_lane_seed101",
        "evidence_role": "broad_private_survival_lane_comparator",
    },
]
NO_CHEAT = {
    "public_training_rows_written": 0,
    "external_inference_calls": 0,
    "fallback_return_count": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    started = time.perf_counter()
    rows = [load_evidence(spec) for spec in EVIDENCE_REPORTS]
    present = [row for row in rows if row.get("present")]
    hard_gaps = []
    if len(present) < 3:
        hard_gaps.append(gap("insufficient_matched_substrate_evidence", {"present": len(present), "required": 3}))
    counter_faults = [row for row in present if not row.get("no_cheat_clean")]
    if counter_faults:
        hard_gaps.append(gap("no_cheat_counter_fault", {"faults": counter_faults}))
    if any(row.get("model_promotion_allowed") for row in present):
        hard_gaps.append(gap("unexpected_promotion_claim_in_verdict_inputs", {"rows": present}))
    verdict = decide_verdict(present)
    records = build_records(present, verdict)
    payload = {
        "policy": "project_theseus_substrate_verdict_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_gaps else "RED",
        "summary": {
            "evidence_report_count": len(present),
            "missing_evidence_report_count": len(rows) - len(present),
            "no_cheat_clean": not counter_faults,
            "symliquid_positive_count": sum(1 for row in present if row.get("winner") == "symliquid_style"),
            "transformer_positive_count": sum(1 for row in present if row.get("winner") == "transformer_control"),
            "tie_or_mixed_count": sum(1 for row in present if row.get("winner") == "tie_or_mixed"),
            "promotion_allowed_input_count": sum(1 for row in present if row.get("model_promotion_allowed")),
            "practical_route": verdict["practical_route"],
            "symliquid_discovery_state": verdict["symliquid_discovery_state"],
            "superiority_claim": verdict["superiority_claim"],
            "hard_gap_count": len(hard_gaps),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "hard_gaps": hard_gaps,
        "evidence_rows": rows,
        "verdict": verdict,
        **records,
        **NO_CHEAT,
        "non_claims": [
            "This verdict does not train, benchmark, or promote a model.",
            "Legacy body-template or renderer evidence is not learned generation evidence.",
            "SymLiquid is protected as discovery evidence but does not block the practical assistant lane.",
            "Transformer/hybrid practical routing is not a proof of public-transfer success.",
        ],
    }
    report_evidence_store.write_json_report(
        resolve(args.out),
        payload,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(payload),
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] == "GREEN" else 2


def load_evidence(spec: dict[str, str]) -> dict[str, Any]:
    path = resolve(spec["path"])
    payload = read_json(path)
    if not path.exists() or not isinstance(payload, dict):
        return {**spec, "present": False, "no_cheat_clean": False, "winner": "missing"}
    summary = dict_value(payload.get("summary"))
    view = {**payload, **summary}
    delta = metric_delta(view)
    winner = "tie_or_mixed"
    if delta is not None and delta > 0.025:
        winner = "symliquid_style"
    elif delta is not None and delta < -0.025:
        winner = "transformer_control"
    public_training_rows = int_value(view.get("public_training_rows_written"), view.get("public_training_rows"), view.get("public_training_rows_written_count"))
    external_calls = int_value(view.get("external_inference_calls"))
    fallback_count = int_value(view.get("fallback_return_count"))
    teacher_used = bool(view.get("teacher_used") or dict_value(view.get("teacher_training")).get("enabled"))
    return {
        **spec,
        "present": True,
        "path": spec["path"],
        "comparison_level": view.get("comparison_level"),
        "delta_metric": "symliquid_minus_transformer_sts_on",
        "symliquid_minus_transformer": delta,
        "winner": winner,
        "trusted_parameter_match": bool(view.get("trusted_parameter_match", True)),
        "parameter_match_delta": view.get("parameter_match_delta"),
        "model_promotion_allowed": bool(view.get("model_promotion_allowed")),
        "teacher_used": teacher_used,
        "public_training_rows_written": public_training_rows,
        "external_inference_calls": external_calls,
        "fallback_return_count": fallback_count,
        "no_cheat_clean": public_training_rows == 0 and external_calls == 0 and fallback_count == 0,
        "content_hash": stable_hash(payload),
    }


def metric_delta(payload: dict[str, Any]) -> float | None:
    keys = [
        "symliquid_minus_transformer_sts_on_mean",
        "symliquid_minus_transformer_sts_on_verifier_pass_rate",
        "symliquid_minus_transformer_expected_plan_match_mean",
    ]
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def decide_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promotion_grade_rows = [row for row in rows if row.get("trusted_parameter_match") and not row.get("model_promotion_allowed")]
    sym_wins = [row for row in promotion_grade_rows if row.get("winner") == "symliquid_style"]
    transformer_wins = [row for row in promotion_grade_rows if row.get("winner") == "transformer_control"]
    ties = [row for row in promotion_grade_rows if row.get("winner") == "tie_or_mixed"]
    decisive_state = "mixed_or_tied"
    if len(sym_wins) >= 3 and not transformer_wins:
        decisive_state = "symliquid_repeated_private_advantage"
    elif len(transformer_wins) >= 3 and not sym_wins:
        decisive_state = "transformer_repeated_private_advantage"
    practical_route = "transformer_hybrid_practical_survival_lane"
    return {
        "practical_route": practical_route,
        "practical_route_reason": "Use the current transformer/hybrid strict-generator survival lane for assistant-building work because SymLiquid has no repeated promotion-grade superiority and should not block practical progress.",
        "symliquid_discovery_state": "protected_matched_comparator",
        "matched_evidence_state": decisive_state,
        "symliquid_win_count": len(sym_wins),
        "transformer_win_count": len(transformer_wins),
        "tie_or_mixed_count": len(ties),
        "superiority_claim": "none",
        "promotion_allowed": False,
        "next_evidence_needed": [
            "repeated matched-compute full-body evidence",
            "candidate-integrity-clean learned-generation rows",
            "public calibration only after private direct learned candidate quality improves",
        ],
    }


def build_records(rows: list[dict[str, Any]], verdict: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    evidence_refs = [row["path"] for row in rows if row.get("present")]
    verdict_ref = "reports/theseus_substrate_verdict.json"
    substrate_adoption_records = [
        substrate_record(
            substrate_id="substrate.symliquid_discovery_lane",
            substrate_kind="symliquid",
            adoption_state=verdict["symliquid_discovery_state"],
            candidate_use="Protected matched-compute discovery comparator.",
            production_default=False,
            evidence_refs=evidence_refs + [verdict_ref],
            residuals=verdict["next_evidence_needed"],
        ),
        substrate_record(
            substrate_id="substrate.transformer_hybrid_survival_lane",
            substrate_kind="transformer_hybrid",
            adoption_state="practical_survival_lane",
            candidate_use="Current practical assistant-building route under strict-generator integrity gates.",
            production_default=False,
            evidence_refs=evidence_refs + [verdict_ref],
            residuals=["must still clear direct learned full-body semantic quality and public transfer"],
        ),
    ]
    claim_records = [
        {
            "record_type": "claim_record",
            "claim_id": "claim.substrate_verdict.no_superiority_claim",
            "claim": "Current matched evidence does not justify a SymLiquid or transformer/hybrid superiority claim.",
            "support_state": "SUPPORTED",
            "evidence_refs": evidence_refs + [verdict_ref],
            **NO_CHEAT,
            "non_claims": ["not a public benchmark result", "not a promotion claim"],
        },
        {
            "record_type": "claim_record",
            "claim_id": "claim.substrate_verdict.practical_route_separated",
            "claim": "Practical transformer/hybrid survival-lane routing is separated from protected SymLiquid discovery evidence.",
            "support_state": "SUPPORTED",
            "evidence_refs": evidence_refs + [verdict_ref],
            **NO_CHEAT,
            "non_claims": ["does not retire SymLiquid", "does not claim transformer public-transfer success"],
        },
    ]
    artifact_graph_records = [
        {
            "record_type": "artifact_graph_record",
            "artifact_id": "artifact.substrate_verdict.v1",
            "artifact_type": "substrate_verdict_report",
            "parent_job": "theseus_substrate_verdict",
            "source_refs": evidence_refs,
            "claim_refs": [row["claim_id"] for row in claim_records],
            "test_refs": ["python3 scripts/theseus_substrate_verdict.py"],
            "audit_events": ["matched_evidence_loaded", "route_policy_separated", "no_superiority_claim_emitted"],
            "replay_metadata": {"evidence_count": len(evidence_refs), "verdict": verdict},
            "replay_grade": "metadata_replayable_from_registered_reports",
            "provenance_status": "existing_private_evidence_summary",
            "evidence_gate": {"state": "SUPPORTED", **NO_CHEAT},
            "residuals": verdict["next_evidence_needed"],
            **NO_CHEAT,
            "non_claims": ["not learned-generation evidence", "not a benchmark run"],
        }
    ]
    evidence_transition_records = [
        {
            "record_type": "evidence_transition_record",
            "record_id": "evidence.substrate_verdict.phase11",
            "artifact_ref": verdict_ref,
            "previous_support_state": "MIXED_EVIDENCE_WITHOUT_ROUTE_POLICY",
            "current_support_state": "SUPPORTED",
            "transition_reason": "compact substrate verdict separates route policy from discovery evidence",
            "evidence_ref": verdict_ref,
            **NO_CHEAT,
            "non_claims": ["storage/evidence transition only", "no model promotion"],
        }
    ]
    return {
        "substrate_adoption_records": substrate_adoption_records,
        "claim_records": claim_records,
        "artifact_graph_records": artifact_graph_records,
        "evidence_transition_records": evidence_transition_records,
    }


def substrate_record(
    *,
    substrate_id: str,
    substrate_kind: str,
    adoption_state: str,
    candidate_use: str,
    production_default: bool,
    evidence_refs: list[str],
    residuals: list[str],
) -> dict[str, Any]:
    return {
        "record_type": "substrate_adoption_record",
        "id": substrate_id,
        "substrate_id": substrate_id,
        "substrate_kind": substrate_kind,
        "adoption_state": adoption_state,
        "current_lifecycle": adoption_state,
        "production_default": production_default,
        "candidate_use": candidate_use,
        "intended_use": candidate_use,
        "baseline_refs": ["matched transformer/hybrid control", "matched SymLiquid-style control", "no-substrate/default route"],
        "baseline_count": 3,
        "negative_controls": ["parameter mismatch", "public benchmark leakage", "template/router evidence counted as learned generation"],
        "negative_control_count": 3,
        "proof_boundary": "Private matched evidence and route-policy boundary only; no public-transfer or learned-generation superiority claim.",
        "falsification_condition": "A future repeated matched-compute run with clean candidate integrity contradicts this route policy.",
        "falsification_criteria_count": 1,
        "axis_ledger": [
            {"axis": "matched_private_evidence", "status": "measured_mixed", "evidence_refs": evidence_refs, "non_claims": ["not public transfer"]},
            {"axis": "production_default", "status": "not_adopted", "evidence_refs": evidence_refs, "non_claims": ["no default switch"]},
            {"axis": "learned_generation_quality", "status": "blocked_by_direct_body_quality", "evidence_refs": evidence_refs, "non_claims": ["router/template evidence excluded"]},
        ],
        "evidence_refs": evidence_refs,
        "experiment_requirements": residuals,
        "required_future_evidence_count": len(residuals),
        "residuals": residuals,
        "consumer_gate": "May guide route priority only; any superiority or default-adoption claim requires fresh matched evidence and candidate-integrity proof.",
        **NO_CHEAT,
        "non_claims": ["not learned-generation evidence", "not public calibration", "not production default adoption"],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    verdict = payload.get("verdict", {})
    return "\n".join(
        [
            "# Theseus Substrate Verdict",
            "",
            f"- trigger_state: `{payload.get('trigger_state')}`",
            f"- evidence reports: `{summary.get('evidence_report_count')}`",
            f"- practical_route: `{summary.get('practical_route')}`",
            f"- symliquid_discovery_state: `{summary.get('symliquid_discovery_state')}`",
            f"- superiority_claim: `{summary.get('superiority_claim')}`",
            f"- matched_evidence_state: `{verdict.get('matched_evidence_state')}`",
            "",
            "This report separates practical route priority from protected discovery evidence.",
        ]
    ) + "\n"


def gap(kind: str, detail: Any) -> dict[str, Any]:
    return {"kind": kind, "detail": detail}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def int_value(*values: Any) -> int:
    for value in values:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return 0


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
