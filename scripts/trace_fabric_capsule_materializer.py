"""Materialize admitted trace-fabric capsules into governed training rows.

This is the second half of the MoECOT trace-fabric port. Admission keeps
capsules metadata-only; materialization turns accepted metadata into bounded
local training packets without copying raw traces, hidden benchmark answers, or
provider outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADMISSION = ROOT / "reports" / "trace_fabric_capsule_admission.json"
DEFAULT_CANDIDATES = ROOT / "data" / "training_sources" / "trace_fabric_capsule_candidates.jsonl"
DEFAULT_OUT = ROOT / "reports" / "trace_fabric_capsule_materialization.json"
DEFAULT_ROWS_OUT = ROOT / "data" / "training_sources" / "trace_fabric_materialized_training_rows.jsonl"

ANSWER_KEY_BLOCKLIST = {
    "answer",
    "answers",
    "expected_answer",
    "expected_answers",
    "gold",
    "gold_answer",
    "ground_truth",
    "label",
    "labels",
    "oracle_answer",
    "reference_answer",
    "reference_answers",
    "solution",
    "solutions",
}
RAW_PAYLOAD_KEY_BLOCKLIST = {
    "action_trace",
    "actions",
    "messages",
    "observation",
    "observations",
    "payload",
    "raw",
    "raw_payload",
    "step_receipts",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", default=str(DEFAULT_ADMISSION.relative_to(ROOT)))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--rows-out", default=str(DEFAULT_ROWS_OUT.relative_to(ROOT)))
    parser.add_argument("--max-rows", type=int, default=64)
    args = parser.parse_args()

    admission_path = resolve(args.admission)
    candidates_path = resolve(args.candidates)
    rows_out = resolve(args.rows_out)
    admission = read_json(admission_path)
    candidates = read_jsonl(candidates_path)
    if not candidates:
        candidates = [row for row in admission.get("accepted_candidate_rows", []) if isinstance(row, dict)]
    report = build_report(
        admission=admission,
        admission_path=admission_path,
        candidates=candidates,
        candidates_path=candidates_path,
        rows_out=rows_out,
        max_rows=max(1, args.max_rows),
    )
    write_jsonl(rows_out, report["materialized_rows"])
    report["rows_path"] = rel(rows_out)
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(
    *,
    admission: dict[str, Any],
    admission_path: Path,
    candidates: list[dict[str, Any]],
    candidates_path: Path,
    rows_out: Path,
    max_rows: int,
) -> dict[str, Any]:
    materialized: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for candidate in candidates:
        if len(materialized) >= max_rows:
            audits.append(audit(candidate, "skipped_max_rows_reached", ["max_rows_reached"]))
            continue
        row, audit_row = materialize_candidate(candidate)
        audits.append(audit_row)
        if row:
            materialized.append(row)

    rejections = Counter(reason for row in audits for reason in row.get("reasons", []))
    lane_counts = Counter(str(row.get("lane") or "unknown") for row in materialized)
    source_kinds = Counter(str(row.get("source_kind") or "unknown") for row in materialized)
    raw_payload_rows = [row.get("row_id") for row in materialized if row.get("raw_payload_copied")]
    gates = [
        gate("admission_report_present", bool(admission), rel_or_abs(admission_path)),
        gate("admission_green_or_yellow", admission.get("trigger_state") in {"GREEN", "YELLOW"}, admission.get("trigger_state")),
        gate("candidates_present", len(candidates) > 0, f"candidates={len(candidates)} path={rel_or_abs(candidates_path)}"),
        gate("materialized_rows_present", len(materialized) > 0, f"rows={len(materialized)} path={rel(rows_out)}"),
        gate("bounded_row_count", len(materialized) <= max_rows, f"rows={len(materialized)} max={max_rows}"),
        gate("source_hashes_verified", rejections.get("source_sha256_mismatch", 0) == 0, dict(rejections)),
        gate("no_answer_key_leaks", rejections.get("answer_like_key_present", 0) == 0, dict(rejections)),
        gate("no_raw_payload_rows", not raw_payload_rows, raw_payload_rows[:20]),
        gate("external_inference_zero", all(int_or(row.get("external_inference_calls")) == 0 for row in materialized), "materializer local metadata only"),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    if not materialized:
        trigger_state = "RED"
    return {
        "policy": "project_theseus_trace_fabric_capsule_materializer_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "admission_report": rel_or_abs(admission_path),
        "candidates_path": rel_or_abs(candidates_path),
        "rows_path": rel(rows_out),
        "summary": {
            "candidates": len(candidates),
            "materialized_rows": len(materialized),
            "max_rows": max_rows,
            "lane_counts": dict(lane_counts),
            "source_kind_counts": dict(source_kinds),
            "rejections": dict(rejections),
            "raw_payload_rows": len(raw_payload_rows),
            "external_inference_calls": 0,
        },
        "materialized_rows": materialized,
        "candidate_audits": audits,
        "gates": gates,
        "hard_invariants": [
            "Only accepted trace-fabric candidates are materialized.",
            "Raw trace files referenced by source reports are never opened by this materializer.",
            "Rows contain sanitized metadata, residuals, hashed receipts, and governance targets only.",
            "Reports containing answer-like keys are quarantined instead of becoming training rows.",
            "Materialized rows are not public benchmark score evidence.",
        ],
        "external_inference_calls": 0,
    }


def materialize_candidate(candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    reasons: list[str] = []
    source_path = resolve(str(candidate.get("source_path") or ""))
    if candidate.get("source_type") != "trace_fabric_capsule":
        reasons.append("not_trace_fabric_candidate")
    if candidate.get("training_use_state") != "candidate_metadata_only_pending_exporter":
        reasons.append("not_pending_exporter")
    if candidate.get("raw_payload_copied"):
        reasons.append("raw_payload_already_copied")
    if not candidate.get("not_public_benchmark_claim_evidence"):
        reasons.append("missing_not_public_benchmark_claim_marker")
    if "holdout" not in str(candidate.get("contamination_boundary") or "").lower():
        reasons.append("missing_holdout_boundary")
    if not source_path.exists():
        reasons.append("source_missing")
        return None, audit(candidate, "rejected", reasons)
    expected_sha = str(candidate.get("source_sha256") or "")
    actual_sha = sha256_file(source_path)
    if expected_sha and actual_sha != expected_sha:
        reasons.append("source_sha256_mismatch")
    source = read_json(source_path)
    if not source:
        reasons.append("source_report_not_json_object")
    if any_answer_key(source):
        reasons.append("answer_like_key_present")
    if int_or(source.get("external_inference_calls")) != 0:
        reasons.append("source_external_inference_nonzero")
    sanitized = sanitize_source_report(source)
    if raw_payload_key_hits(sanitized):
        reasons.append("sanitized_raw_payload_key_present")
    if reasons:
        return None, audit(candidate, "rejected", reasons, actual_sha=actual_sha)

    capsule_id = str(candidate.get("capsule_id") or stable_hash(candidate)[:16])
    lane = lane_for(candidate, source)
    source_kind = str(source.get("frontier_family") or source.get("runner_family") or candidate.get("trace_kind") or "trace")
    prompt_payload = {
        "capsule_id": capsule_id,
        "trace_kind": candidate.get("trace_kind"),
        "source_kind": source_kind,
        "quality_score": candidate.get("quality_score"),
        "utility_score": candidate.get("utility_score"),
        "sanitized_source": sanitized,
        "constraints": [
            "Do not use this as public benchmark score evidence.",
            "Do not infer from raw traces; only use the sanitized capsule metadata.",
            "Preserve the contamination and retention boundary.",
        ],
    }
    answer_payload = target_answer(candidate, source, sanitized, lane)
    prompt = (
        "Review this governed Theseus trace-fabric capsule and decide the safe training lesson, "
        "residual focus, and next action.\n\n"
        + json.dumps(prompt_payload, sort_keys=True, ensure_ascii=True)
    )
    answer = json.dumps(answer_payload, sort_keys=True, ensure_ascii=True)
    row_id = "trace_capsule_" + stable_hash({"capsule_id": capsule_id, "source_sha": actual_sha, "answer": answer})[:16]
    row = {
        "row_id": row_id,
        "dataset_id": "dataset.trace_fabric_capsule_materialized.v1",
        "source_type": "trace_fabric_capsule_materialized",
        "source_kind": source_kind,
        "split": "train",
        "lane": lane,
        "task_family": "trace_capsule_governance",
        "prompt": prompt,
        "answer": answer,
        "prompt_sha256": sha256_text(prompt),
        "answer_sha256": sha256_text(answer),
        "capsule_id": capsule_id,
        "trace_kind": candidate.get("trace_kind"),
        "source_path": rel_or_abs(source_path),
        "source_sha256": actual_sha,
        "quality_score": float_or(candidate.get("quality_score")),
        "utility_score": float_or(candidate.get("utility_score")),
        "raw_payload_copied": False,
        "raw_trace_opened": False,
        "not_public_benchmark_claim_evidence": True,
        "holdout_boundary_checked": True,
        "reference_answers_present": False,
        "external_inference_calls": 0,
        "created_utc": now(),
    }
    return row, audit(candidate, "materialized", [], actual_sha=actual_sha, row_id=row_id)


def sanitize_source_report(source: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for item in source.get("checks") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("gate") or "")[:96]
        evidence = item.get("evidence")
        checks.append(
            {
                "name": name,
                "passed": bool(item.get("passed")),
                "evidence_sha256": stable_hash(evidence)[:16],
            }
        )
    residuals = []
    for item in source.get("residuals") or []:
        if not isinstance(item, dict):
            continue
        residuals.append(
            {
                "type": str(item.get("type") or "residual")[:80],
                "detail": str(item.get("detail") or "")[:220],
            }
        )
    metrics = source.get("metrics") if isinstance(source.get("metrics"), dict) else {}
    safe_metrics: dict[str, Any] = {}
    for key in [
        "source_present",
        "manifest_count",
        "task_or_harness_count",
        "node_available",
        "bun_available",
        "docker_available",
        "podman_available",
        "container_runtime_available",
        "steps",
        "total_reward",
        "survival_ratio",
    ]:
        if key in metrics and is_scalar(metrics[key]):
            safe_metrics[key] = metrics[key]
    evaluation = metrics.get("evaluation") if isinstance(metrics.get("evaluation"), dict) else {}
    safe_evaluation = {
        key: value
        for key, value in evaluation.items()
        if key
        in {
            "score",
            "steps",
            "expected_steps",
            "total_reward",
            "reward_per_step",
            "survival_ratio",
            "termination_rate",
            "mean_altitude_error",
            "mean_xy_error",
            "mean_target_distance",
            "mean_abs_vz",
            "mean_action_delta",
            "trace_rows",
        }
        and is_scalar(value)
    }
    if safe_evaluation:
        safe_metrics["evaluation"] = safe_evaluation
    budget = source.get("budget") if isinstance(source.get("budget"), dict) else {}
    safe_budget = {
        key: value
        for key, value in budget.items()
        if key
        in {
            "episodes",
            "steps",
            "train_iterations",
            "train_population",
            "elite_count",
            "eval_seed_count",
            "train_candidate_evaluations",
            "train_env_steps_budget",
        }
        and is_scalar(value)
    }
    return {
        "methodology": source.get("methodology"),
        "frontier_family": source.get("frontier_family"),
        "card_id": source.get("card_id"),
        "runner_family": source.get("runner_family"),
        "seed": source.get("seed"),
        "episodes": source.get("episodes"),
        "steps": source.get("steps"),
        "status": source.get("status"),
        "summary": compact_summary(source.get("summary")),
        "safe_metrics": safe_metrics,
        "budget": safe_budget,
        "residuals": residuals[:12],
        "checks": checks[:16],
        "permission_envelope": source.get("permission_envelope") if isinstance(source.get("permission_envelope"), dict) else {},
        "external_inference_calls": int_or(source.get("external_inference_calls")),
    }


def target_answer(candidate: dict[str, Any], source: dict[str, Any], sanitized: dict[str, Any], lane: str) -> dict[str, Any]:
    residuals = sanitized.get("residuals") or []
    first_residual = residuals[0] if residuals else {"type": "none", "detail": "no residual recorded"}
    score = get_path(source, ["summary", "score"], get_path(source, ["summary", "accuracy"], None))
    return {
        "training_use_state": "materialized_trace_capsule_training_row",
        "lane": lane,
        "capsule_id": candidate.get("capsule_id"),
        "decision": "use_as_governed_local_training_packet",
        "not_public_benchmark_claim_evidence": True,
        "raw_payload_policy": "raw_trace_not_opened_or_copied",
        "holdout_policy": "answer_like_keys_absent_and_reference_answers_not_materialized",
        "external_inference_policy": "external_inference_zero_required",
        "quality_utility": {
            "quality_score": candidate.get("quality_score"),
            "utility_score": candidate.get("utility_score"),
            "source_score": score,
        },
        "residual_focus": first_residual,
        "next_action": next_action_for(lane, first_residual, sanitized),
    }


def next_action_for(lane: str, residual: dict[str, Any], sanitized: dict[str, Any]) -> str:
    residual_type = str(residual.get("type") or "").lower()
    if "drone" in lane:
        if "waypoint" in residual_type:
            return "add waypoint-gate reward shaping and replay the sim-only smoke before any hardware lane"
        if "survival" in residual_type or "altitude" in residual_type:
            return "tighten stabilizing controller curriculum and keep hardware gates closed"
        return "use the capsule as sim-only drone residual evidence with blackbox parity still required"
    if "code" in lane:
        if "container" in residual_type:
            return "wire a local container runtime or use source-present task-contract pressure only"
        if "adapter" in residual_type or "harness" in residual_type:
            return "map the source harness into a deterministic local endpoint/task adapter"
        return "use the capsule to train code-agent residual triage and transfer-artifact selection"
    return "use the capsule for autonomy trace governance and require a fresh report before promotion"


def lane_for(candidate: dict[str, Any], source: dict[str, Any]) -> str:
    text = " ".join(
        str(value or "")
        for value in [
            candidate.get("trace_kind"),
            source.get("frontier_family"),
            source.get("runner_family"),
            source.get("card_id"),
            source.get("card_name"),
        ]
    ).lower()
    if "drone" in text or "pyflyt" in text:
        return "rl_environment_trace_governance"
    if any(token in text for token in ["code", "swe", "terminal", "repo"]):
        return "code_agent_trace_governance"
    if "teacher" in text or "self_edit" in text:
        return "self_evolution_trace_governance"
    return "autonomy_trace_governance"


def any_answer_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in ANSWER_KEY_BLOCKLIST:
                return True
            if any_answer_key(child):
                return True
    elif isinstance(value, list):
        return any(any_answer_key(item) for item in value)
    return False


def raw_payload_key_hits(value: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in RAW_PAYLOAD_KEY_BLOCKLIST:
                hits.append(str(key))
            hits.extend(raw_payload_key_hits(child))
    elif isinstance(value, list):
        for item in value:
            hits.extend(raw_payload_key_hits(item))
    return hits


def audit(candidate: dict[str, Any], state: str, reasons: list[str], *, actual_sha: str = "", row_id: str = "") -> dict[str, Any]:
    return {
        "capsule_id": candidate.get("capsule_id"),
        "source_path": candidate.get("source_path"),
        "state": state,
        "reasons": reasons,
        "expected_sha256": candidate.get("source_sha256"),
        "actual_sha256": actual_sha,
        "row_id": row_id,
    }


def compact_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keep = {}
    for key in ["suite", "accuracy", "score", "total_tool_calls"]:
        if key in value and is_scalar(value[key]):
            keep[key] = value[key]
    return keep


def is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def get_path(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")).hexdigest()


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
