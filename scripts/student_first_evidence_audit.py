"""Student-first evidence audit for Project Theseus.

This report keeps the central promise honest: public-code progress can only be
claimed when candidates come from token-level student generation. Deterministic
helpers, rankers, loop-closure tools, and verifier layers may support or reject
candidates, but they cannot be counted as proof that the student learned.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from real_code_benchmark_runtime import benchmark_candidate_eligible, normalize_student_candidate
from candidate_integrity import recompute_candidate_integrity
from theseus_archive_resolver import read_jsonl_follow_pointer


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_CODE_FLOOR = 0.70
DEFAULT_REAL_CODE_REPORT = "reports/real_code_benchmark_graduation.json"
DEFAULT_CANDIDATE_MANIFEST = "reports/student_code_candidates.jsonl"
VALID_STUDENT_SOURCES = {
    "student_code_lm_checkpoint_v1",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-code", default="reports/real_code_benchmark_graduation.json")
    parser.add_argument("--candidate-manifest", default="reports/student_code_candidates.jsonl")
    parser.add_argument("--code-lm", default="reports/code_lm_closure.json")
    parser.add_argument("--student-learning", default="reports/student_learning_closure.json")
    parser.add_argument("--candidate-gate", default="reports/candidate_promotion_gate.json")
    parser.add_argument("--broad-matrix", default="reports/broad_transfer_matrix.json")
    parser.add_argument("--out", default="reports/student_first_evidence_audit.json")
    parser.add_argument("--markdown-out", default="reports/student_first_evidence_audit.md")
    args = parser.parse_args()

    real_code, real_code_path = resolve_real_code_report(args.real_code)
    code_lm = read_json(resolve(args.code_lm), {})
    student_learning = read_json(resolve(args.student_learning), {})
    candidate_gate = read_json(resolve(args.candidate_gate), {})
    broad_matrix, broad_matrix_path = resolve_broad_matrix(args.broad_matrix, real_code_path)
    candidate_manifest_path = resolve_candidate_manifest(args.candidate_manifest, real_code)
    candidates = read_jsonl(candidate_manifest_path)
    manifest = candidate_manifest_summary(candidates)
    summary = object_field(real_code, "summary")
    candidate_source = str(real_code.get("candidate_source") or "")
    public_rate = number(summary.get("real_public_task_pass_rate"))
    public_claim = str(real_code.get("public_benchmark_score_claim") or "")
    token_student_valid = bool(
        candidate_source in VALID_STUDENT_SOURCES
        and public_claim.endswith("_public_task_calibration_only")
        and bool(summary.get("token_level_code_generation_learned"))
        and bool(summary.get("student_candidate_benchmark_integrity_valid"))
        and int(summary.get("functional_promotion_count") or 0) > 0
        and int(summary.get("full_body_token_candidate_count") or 0) > 0
        and int(summary.get("grammar_masked_learned_token_candidate_count") or 0) > 0
        and int(summary.get("benchmark_promotion_eligible_candidate_count") or 0) > 0
        and int(summary.get("candidate_integrity_mismatch_count") or 0) == 0
        and int(summary.get("integrity_verified_candidate_count") or 0) > 0
        and int(summary.get("template_like_candidate_count") or 0) == 0
        and int(summary.get("loop_closure_candidate_count") or 0) == 0
        and int(summary.get("expression_memory_fallback_count") or 0) == 0
        and int(real_code.get("external_inference_calls") or 0) == 0
    )
    ranker_reported_as_learning = bool(
        student_learning.get("policy")
        and get_path(student_learning, ["summary", "token_level_code_generation_learned"], False)
    )
    candidate_promote = bool(candidate_gate.get("promote"))
    gates = [
        gate("real_code_report_present", real_code.get("policy") == "project_theseus_real_code_benchmark_graduation_v1", real_code.get("policy")),
        gate("public_score_claim_quarantined", public_claim.endswith("_public_task_calibration_only"), public_claim),
        gate("candidate_source_is_code_lm_checkpoint", candidate_source in VALID_STUDENT_SOURCES, candidate_source),
        gate("token_level_student_generation_valid", token_student_valid, evidence_bundle(real_code, manifest)),
        gate("candidate_manifest_has_full_body_token_candidates", manifest["full_body_token_candidate_count"] > 0, manifest),
        gate("candidate_manifest_has_grammar_masked_learned_token_candidates", manifest["grammar_masked_learned_token_candidate_count"] > 0, manifest),
        gate("candidate_manifest_has_promotion_eligible_full_body_candidates", manifest["benchmark_promotion_eligible_candidate_count"] > 0, manifest),
        gate("candidate_manifest_recomputed_integrity_clean", manifest["candidate_integrity_mismatch_count"] == 0 and manifest["integrity_verified_candidate_count"] > 0, manifest),
        gate(
            "functional_promotion_requires_behavioral_pass",
            int(summary.get("functional_promotion_count") or 0) > 0,
            {
                "functional_promotion_count": summary.get("functional_promotion_count"),
                "functional_promotion_fraction": summary.get("functional_promotion_fraction"),
                "functional_promotion_rate_ci95": summary.get("functional_promotion_rate_ci95"),
            },
        ),
        gate("candidate_manifest_has_no_templates_or_tools", manifest["template_like_candidate_count"] == 0 and manifest["loop_closure_candidate_count"] == 0, manifest),
        gate("ranker_not_counted_as_token_learning", not ranker_reported_as_learning, get_path(student_learning, ["summary"], {})),
        gate("promotion_requires_floor", (not candidate_promote) or public_rate >= PUBLIC_CODE_FLOOR, f"promote={candidate_promote} public_rate={public_rate} floor={PUBLIC_CODE_FLOOR}"),
        gate("external_inference_zero", int(real_code.get("external_inference_calls") or 0) == 0, real_code.get("external_inference_calls")),
    ]
    broad_audit = broad_matrix_audit(broad_matrix, broad_matrix_path)
    gates.extend(
        [
            gate("broad_matrix_present", broad_audit["present"], broad_audit["path"]),
            gate("broad_dirty_evidence_quarantined", broad_audit["dirty_evidence_quarantined"], broad_audit),
            gate("broad_loader_only_separated", broad_audit["loader_only_separated"], broad_audit["loader_only_cards"]),
            gate("broad_below_floor_labeled", broad_audit["below_floor_labeled"], broad_audit["below_floor_cards"]),
        ]
    )
    hard_failures = [row["gate"] for row in gates if not row["passed"] and row["gate"] not in {"promotion_requires_floor"}]
    trigger_state = "GREEN" if not hard_failures else "RED"
    if trigger_state == "GREEN" and public_rate < PUBLIC_CODE_FLOOR:
        trigger_state = "YELLOW"
    if trigger_state == "GREEN" and broad_audit["warning_count"] > 0:
        trigger_state = "YELLOW"
    payload = {
        "policy": "project_theseus_student_first_evidence_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "thesis": "Only token-level student generation can be cited as learned public-code transfer.",
        "summary": {
            "real_code_report": real_code_path,
            "candidate_manifest": display_path(candidate_manifest_path),
            "candidate_source": candidate_source,
            "public_benchmark_score_claim": public_claim,
            "public_task_pass_rate": public_rate,
            "required_public_task_floor": PUBLIC_CODE_FLOOR,
            "floor_gap": round(max(0.0, PUBLIC_CODE_FLOOR - public_rate), 6),
            "student_first_public_transfer_valid": token_student_valid,
            "promotion_allowed_by_evidence": bool(token_student_valid and public_rate >= PUBLIC_CODE_FLOOR),
            "candidate_promote": candidate_promote,
            "code_lm_trigger_state": code_lm.get("trigger_state"),
            "ranker_lane_status": student_learning.get("trigger_state"),
            "ranker_counted_as_token_learning": ranker_reported_as_learning,
            "external_inference_calls": 0,
            "broad_matrix": broad_audit,
            **manifest,
        },
        "gates": gates,
        "score_semantics": "audit only; does not improve score and does not promote candidates",
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if trigger_state in {"GREEN", "YELLOW"} else 2


def candidate_manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [row for row in (normalize_student_candidate(row) for row in rows) if row]
    family_counts: dict[str, int] = {}
    claimed_promotion_by_family: dict[str, int] = {}
    integrity_verified_by_family: dict[str, int] = {}
    mismatch_counts: dict[str, int] = {}
    integrity_verified_count = 0
    mismatch_count = 0
    for row in normalized:
        integrity = row.get("candidate_integrity") if isinstance(row.get("candidate_integrity"), dict) else recompute_candidate_integrity(row)
        family = str(integrity.get("recomputed_candidate_family") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        if bool(integrity.get("self_declared_flags", {}).get("benchmark_promotion_eligible")):
            claimed_promotion_by_family[family] = claimed_promotion_by_family.get(family, 0) + 1
        if integrity_verified(integrity):
            integrity_verified_count += 1
            integrity_verified_by_family[family] = integrity_verified_by_family.get(family, 0) + 1
        for mismatch in integrity.get("integrity_mismatches") or []:
            mismatch = str(mismatch)
            mismatch_count += 1
            mismatch_counts[mismatch] = mismatch_counts.get(mismatch, 0) + 1
    return {
        "candidate_count": len(normalized),
        "raw_candidate_count": len(rows),
        "token_level_candidate_count": sum(1 for row in normalized if truthy(row.get("token_level_code_generation_learned"))),
        "full_body_token_candidate_count": sum(1 for row in normalized if truthy(row.get("full_body_token_candidate"))),
        "grammar_masked_learned_token_candidate_count": sum(1 for row in normalized if truthy(row.get("grammar_masked_learned_token_candidate"))),
        "benchmark_promotion_eligible_candidate_count": sum(1 for row in normalized if benchmark_candidate_eligible(row)),
        "template_like_candidate_count": sum(1 for row in normalized if truthy(row.get("template_like_candidate"))),
        "loop_closure_candidate_count": sum(1 for row in normalized if truthy(row.get("loop_closure_generated"))),
        "expression_memory_fallback_count": sum(1 for row in normalized if truthy(row.get("expression_memory_fallback"))),
        "candidate_integrity_policy": "project_theseus_recomputed_candidate_integrity_v1",
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "claimed_promotion_by_family": dict(sorted(claimed_promotion_by_family.items())),
        "integrity_verified_by_family": dict(sorted(integrity_verified_by_family.items())),
        "candidate_integrity_mismatch_count": mismatch_count,
        "candidate_integrity_mismatch_counts": dict(sorted(mismatch_counts.items())),
        "integrity_verified_candidate_count": integrity_verified_count,
    }


def evidence_bundle(real_code: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    summary = object_field(real_code, "summary")
    return {
        "candidate_source": real_code.get("candidate_source"),
        "score_claim": real_code.get("public_benchmark_score_claim"),
        "token_level": summary.get("token_level_code_generation_learned"),
        "grammar_masked_learned_token": summary.get("grammar_masked_learned_token_candidate_count"),
        "benchmark_integrity": summary.get("student_candidate_benchmark_integrity_valid"),
        "candidate_integrity_mismatch_count": summary.get("candidate_integrity_mismatch_count", manifest.get("candidate_integrity_mismatch_count")),
        "integrity_verified_candidate_count": summary.get(
            "integrity_verified_candidate_count",
            manifest.get("integrity_verified_candidate_count"),
        ),
        "functional_promotion_count": summary.get("functional_promotion_count"),
        "functional_promotion_fraction": summary.get("functional_promotion_fraction"),
        "candidate_family_counts": summary.get("candidate_family_counts", manifest.get("candidate_family_counts")),
        "templates": summary.get("template_like_candidate_count"),
        "loop_closure": summary.get("loop_closure_candidate_count"),
        "manifest": manifest,
    }


def broad_matrix_audit(matrix: dict[str, Any], matrix_path: str) -> dict[str, Any]:
    rows = [row for row in matrix.get("rows", []) if isinstance(row, dict)]
    dirty_rows = [
        {
            "card_id": row.get("card_id"),
            "selected_report": row.get("selected_report"),
            "violations": row.get("no_cheat_violations", []),
            "status": row.get("status"),
        }
        for row in rows
        if not bool(row.get("no_cheat_valid", False))
    ]
    no_clean = [str(item) for item in matrix.get("summary", {}).get("no_clean_student_evidence_cards", [])]
    loader_only = [str(item) for item in matrix.get("summary", {}).get("loader_only_cards", [])]
    below_floor = [str(item) for item in matrix.get("summary", {}).get("cards_below_floor", [])]
    dirty_card_ids = {str(item.get("card_id")) for item in dirty_rows}
    return {
        "present": matrix.get("policy") == "project_theseus_broad_transfer_matrix_v1",
        "path": matrix_path,
        "trigger_state": matrix.get("trigger_state"),
        "public_task_count": get_path(matrix, ["summary", "real_public_task_count"], 0),
        "public_pass_rate": get_path(matrix, ["summary", "real_public_pass_rate"], 0.0),
        "sts_delta": get_path(matrix, ["summary", "real_public_sts_delta"], 0.0),
        "dirty_rows": dirty_rows,
        "dirty_evidence_quarantined": dirty_card_ids.issubset(set(no_clean)),
        "no_clean_cards": no_clean,
        "loader_only_cards": loader_only,
        "below_floor_cards": below_floor,
        "loader_only_separated": all(
            row.get("benchmark_evidence_level") == "public_loader_regression"
            for row in rows
            if str(row.get("card_id")) in set(loader_only)
        ),
        "below_floor_labeled": all(
            "below_public_code_floor" in (row.get("coverage_warnings") or [])
            for row in rows
            if str(row.get("card_id")) in set(below_floor)
        ),
        "warning_count": len(no_clean) + len(loader_only) + len(below_floor),
        "score_semantics": matrix.get("score_semantics"),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    failed = [row["gate"] for row in payload["gates"] if not row["passed"]]
    return "\n".join(
        [
            "# Student-First Evidence Audit",
            "",
            f"Generated: {payload.get('created_utc')}",
            f"Trigger: **{payload.get('trigger_state')}**",
            "",
            f"- Candidate source: {summary.get('candidate_source')}",
            f"- Public pass rate: {summary.get('public_task_pass_rate')} / {summary.get('required_public_task_floor')}",
            f"- Student-first valid: {summary.get('student_first_public_transfer_valid')}",
            f"- Full-body token candidates: {summary.get('full_body_token_candidate_count')}",
            f"- Grammar-masked learned-token candidates: {summary.get('grammar_masked_learned_token_candidate_count')}",
            f"- Templates / loop tools: {summary.get('template_like_candidate_count')} / {summary.get('loop_closure_candidate_count')}",
            f"- Ranker counted as token learning: {summary.get('ranker_counted_as_token_learning')}",
            f"- Broad matrix pass rate: {get_path(summary, ['broad_matrix', 'public_pass_rate'], 'n/a')} over {get_path(summary, ['broad_matrix', 'public_task_count'], 'n/a')} tasks",
            f"- Broad warnings: no-clean={get_path(summary, ['broad_matrix', 'no_clean_cards'], [])} loader-only={get_path(summary, ['broad_matrix', 'loader_only_cards'], [])} below-floor={get_path(summary, ['broad_matrix', 'below_floor_cards'], [])}",
            f"- Failed gates: {', '.join(failed) if failed else 'none'}",
            "",
            "This report is evidence hygiene only. It does not promote a candidate.",
            "",
        ]
    )


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return item if isinstance(item, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def integrity_verified(integrity: dict[str, Any]) -> bool:
    return bool(integrity.get("integrity_verified", integrity.get("promotion_verified", False)))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path, default: Any = None) -> Any:
    default = {} if default is None else default
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return [row for row in read_jsonl_follow_pointer(path) if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return []


def resolve_real_code_report(requested: str) -> tuple[dict[str, Any], str]:
    requested_path = resolve(requested)
    requested_payload = read_json(requested_path, {})
    if requested.replace("\\", "/") != DEFAULT_REAL_CODE_REPORT:
        return requested_payload, display_path(requested_path)
    candidates: list[tuple[int, int, str, str, dict[str, Any]]] = []
    for path in REPORTS.glob("real_code_benchmark_graduation*.json"):
        payload = read_json(path, {})
        if not token_student_valid_report(payload):
            continue
        summary = object_field(payload, "summary")
        candidates.append(
            (
                int(summary.get("public_task_count") or 0),
                int(summary.get("total_case_count") or 0),
                str(payload.get("created_utc") or ""),
                display_path(path),
                payload,
            )
        )
    if not candidates:
        return requested_payload, display_path(requested_path)
    candidates.sort(reverse=True)
    _, _, _, selected_path, selected_payload = candidates[0]
    return selected_payload, selected_path


def resolve_broad_matrix(requested: str, selected_real_code_path: str) -> tuple[dict[str, Any], str]:
    requested_path = resolve(requested)
    requested_payload = read_json(requested_path, {})
    if requested.replace("\\", "/") != "reports/broad_transfer_matrix.json":
        return requested_payload, display_path(requested_path)

    real_path = Path(selected_real_code_path.replace("\\", "/"))
    name = real_path.name
    if name.startswith("real_code_benchmark_graduation_") and name.endswith(".json"):
        slug = name.removeprefix("real_code_benchmark_graduation_").removesuffix(".json")
        matching = REPORTS / f"broad_transfer_matrix_{slug}.json"
        payload = read_json(matching, {})
        if payload.get("policy") == "project_theseus_broad_transfer_matrix_v1":
            return payload, display_path(matching)

    candidates: list[tuple[int, int, str, str, dict[str, Any]]] = []
    for path in REPORTS.glob("broad_transfer_matrix*.json"):
        payload = read_json(path, {})
        if payload.get("policy") != "project_theseus_broad_transfer_matrix_v1":
            continue
        summary = object_field(payload, "summary")
        candidates.append(
            (
                int(summary.get("real_public_task_count") or 0),
                int(summary.get("clean_covered_card_count") or 0),
                str(payload.get("created_utc") or ""),
                display_path(path),
                payload,
            )
        )
    if not candidates:
        return requested_payload, display_path(requested_path)
    candidates.sort(reverse=True)
    _, _, _, selected_path, selected_payload = candidates[0]
    return selected_payload, selected_path


def token_student_valid_report(payload: dict[str, Any]) -> bool:
    summary = object_field(payload, "summary")
    return bool(
        payload.get("policy") == "project_theseus_real_code_benchmark_graduation_v1"
        and payload.get("trigger_state") in {"GREEN", "YELLOW"}
        and payload.get("candidate_source") in VALID_STUDENT_SOURCES
        and str(payload.get("public_benchmark_score_claim") or "").endswith("_public_task_calibration_only")
        and int(summary.get("public_task_count") or 0) > 0
        and bool(summary.get("token_level_code_generation_learned"))
        and bool(summary.get("student_candidate_benchmark_integrity_valid"))
        and int(summary.get("full_body_token_candidate_count") or 0) > 0
        and int(summary.get("grammar_masked_learned_token_candidate_count") or 0) > 0
        and int(summary.get("benchmark_promotion_eligible_candidate_count") or 0) > 0
        and int(summary.get("template_like_candidate_count") or 0) == 0
        and int(summary.get("loop_closure_candidate_count") or 0) == 0
        and int(summary.get("expression_memory_fallback_count") or 0) == 0
        and int(payload.get("external_inference_calls") or 0) == 0
    )


def resolve_candidate_manifest(requested: str, real_code: dict[str, Any]) -> Path:
    if requested.replace("\\", "/") != DEFAULT_CANDIDATE_MANIFEST:
        return resolve(requested)
    manifest = object_field(real_code, "student_candidate_manifest")
    path = str(manifest.get("path") or "").strip()
    return resolve(path) if path else resolve(requested)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
