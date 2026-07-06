"""Admit trace-fabric capsules as governed training-source candidates.

MoECOT's trace-fabric idea is powerful only if raw traces do not slide straight
into training. This script keeps the exchange metadata-only: it scores capsule
admissibility, records source hashes, and writes candidate rows without copying
raw payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_EXCHANGE = ROOT / "reports" / "trace_fabric_training_exchange.json"
DEFAULT_SELF_MOD_PROOF = ROOT / "reports" / "self_mod_proof_bundle.json"
DEFAULT_OUT = ROOT / "reports" / "trace_fabric_capsule_admission.json"
DEFAULT_CANDIDATES_OUT = ROOT / "data" / "training_sources" / "trace_fabric_capsule_candidates.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-exchange", default=str(DEFAULT_TRACE_EXCHANGE.relative_to(ROOT)))
    parser.add_argument("--self-mod-proof", default=str(DEFAULT_SELF_MOD_PROOF.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--candidates-out", default=str(DEFAULT_CANDIDATES_OUT.relative_to(ROOT)))
    parser.add_argument("--min-quality", type=float, default=0.75)
    parser.add_argument("--min-utility", type=float, default=0.70)
    args = parser.parse_args()

    trace_exchange_path = resolve(args.trace_exchange)
    self_mod_proof_path = resolve(args.self_mod_proof)
    trace_exchange = read_json(trace_exchange_path)
    self_mod_proof = read_json(self_mod_proof_path)
    report = build_report(
        trace_exchange=trace_exchange,
        trace_exchange_path=trace_exchange_path,
        self_mod_proof=self_mod_proof,
        self_mod_proof_path=self_mod_proof_path,
        min_quality=args.min_quality,
        min_utility=args.min_utility,
    )
    candidates_out = resolve(args.candidates_out)
    write_jsonl(candidates_out, report["accepted_candidate_rows"])
    report["accepted_candidates_path"] = rel(candidates_out)
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(
    *,
    trace_exchange: dict[str, Any],
    trace_exchange_path: Path,
    self_mod_proof: dict[str, Any],
    self_mod_proof_path: Path,
    min_quality: float,
    min_utility: float,
) -> dict[str, Any]:
    capsules = [row for row in trace_exchange.get("capsules", []) if isinstance(row, dict)]
    proof_ok = self_mod_proof_allows_training(self_mod_proof)
    scored = [score_capsule(row, min_quality, min_utility, proof_ok) for row in capsules]
    accepted = [row for row in scored if row["admission_state"] == "accepted_metadata_only"]
    quarantined = [row for row in scored if row["admission_state"].startswith("quarantined")]
    missing_sources = [
        row
        for row in scored
        if row.get("training_keep") and not row.get("source_exists")
    ]
    raw_payload_keys = [
        row.get("capsule_id")
        for row in capsules
        if any(key in row for key in ["payload", "raw_payload", "messages", "content"])
    ]
    accepted_candidates = [candidate_row(row, trace_exchange_path) for row in accepted]
    teacher_self_edit_accepted = [
        row for row in accepted if str(row.get("trace_kind") or "") == "teacher_self_edit"
    ]
    gates = [
        gate("trace_exchange_report_present", bool(trace_exchange), rel_or_abs(trace_exchange_path)),
        gate("trace_exchange_status_ready", trace_exchange.get("status") in {"READY", "GREEN", "ready"}, trace_exchange.get("status")),
        gate("capsules_present", len(capsules) > 0, f"capsules={len(capsules)}"),
        gate("accepted_capsules_present", len(accepted) > 0, f"accepted={len(accepted)}"),
        gate("accepted_capsule_sources_exist", not any(not row.get("source_exists") for row in accepted), missing_sources[:10]),
        gate("raw_payload_not_embedded_in_exchange", not raw_payload_keys, raw_payload_keys[:20]),
        gate(
            "teacher_self_edit_requires_proof_bundle",
            not teacher_self_edit_accepted or proof_ok,
            f"proof_ok={proof_ok} proof={rel_or_abs(self_mod_proof_path)} accepted_teacher_self_edit={len(teacher_self_edit_accepted)}",
        ),
        gate("external_inference_zero", int_or(trace_exchange.get("external_inference_calls")) == 0, trace_exchange.get("external_inference_calls")),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    if not accepted:
        trigger_state = "RED"
    return {
        "policy": "project_theseus_trace_fabric_capsule_admission_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "source_report": rel_or_abs(trace_exchange_path),
        "self_mod_proof_bundle": rel_or_abs(self_mod_proof_path),
        "thresholds": {"min_quality": min_quality, "min_utility": min_utility},
        "summary": {
            "capsules": len(capsules),
            "accepted_metadata_only": len(accepted),
            "quarantined": len(quarantined),
            "missing_sources": len(missing_sources),
            "raw_payload_key_hits": len(raw_payload_keys),
            "teacher_self_edit_proof_ok": proof_ok,
            "external_inference_calls": int_or(trace_exchange.get("external_inference_calls")),
        },
        "accepted_candidate_rows": accepted_candidates,
        "capsule_admissions": scored,
        "gates": gates,
        "hard_invariants": [
            "Raw trace payloads are not copied by this admission layer.",
            "Training candidates contain hashes, provenance, scores, and policy markers only.",
            "Teacher self-edit traces require a proof bundle before acceptance.",
            "Capsules with missing source files, low quality, low utility, or disabled training_keep are quarantined.",
        ],
        "external_inference_calls": 0,
    }


def score_capsule(row: dict[str, Any], min_quality: float, min_utility: float, proof_ok: bool) -> dict[str, Any]:
    source_path = resolve(str(row.get("path") or ""))
    trace_kind = str(row.get("trace_kind") or "unknown")
    quality = float_or(row.get("quality_score"))
    utility = float_or(row.get("utility_score"))
    training_keep = bool(row.get("training_keep"))
    raw_retention = str(row.get("raw_retention") or "")
    reasons = []
    if not training_keep:
        reasons.append("training_keep_false")
    if quality < min_quality:
        reasons.append(f"quality_below_{min_quality}")
    if utility < min_utility:
        reasons.append(f"utility_below_{min_utility}")
    if raw_retention not in {"bounded_local", "pruned", "redacted_local"}:
        reasons.append("raw_retention_not_bounded")
    if not source_path.exists():
        reasons.append("source_missing")
    if trace_kind == "teacher_self_edit" and not proof_ok:
        reasons.append("teacher_self_edit_proof_missing")
    admission_state = "accepted_metadata_only" if not reasons else "quarantined_" + reasons[0]
    source_sha256 = sha256_file(source_path) if source_path.exists() else ""
    return {
        "capsule_id": row.get("capsule_id"),
        "trace_kind": trace_kind,
        "source_path": rel_or_abs(source_path),
        "source_exists": source_path.exists(),
        "source_sha256": source_sha256,
        "quality_score": quality,
        "utility_score": utility,
        "raw_retention": raw_retention,
        "training_keep": training_keep,
        "admission_state": admission_state,
        "quarantine_reasons": reasons,
        "raw_payload_copied": False,
        "external_inference_calls": 0,
    }


def candidate_row(row: dict[str, Any], trace_exchange_path: Path) -> dict[str, Any]:
    return {
        "source_type": "trace_fabric_capsule",
        "capsule_id": row.get("capsule_id"),
        "trace_kind": row.get("trace_kind"),
        "source_path": row.get("source_path"),
        "source_sha256": row.get("source_sha256"),
        "quality_score": row.get("quality_score"),
        "utility_score": row.get("utility_score"),
        "raw_retention": row.get("raw_retention"),
        "raw_payload_copied": False,
        "source_report": rel_or_abs(trace_exchange_path),
        "training_use_state": "candidate_metadata_only_pending_exporter",
        "contamination_boundary": "must_pass_capsule_level_holdout_and_redaction_checks_before_rows_are_materialized",
        "not_public_benchmark_claim_evidence": True,
        "created_utc": now(),
    }


def self_mod_proof_allows_training(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status") or payload.get("trigger_state") or "").upper()
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    proof_ready = payload.get("proof_ready")
    if proof_ready is True:
        return True
    if status in {"READY", "GREEN"} and int_or(summary.get("failed_checks")) == 0:
        return True
    return False


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
