#!/usr/bin/env python3
"""Independently qualify the standard causal-transformer survival evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "standard_causal_transformer_survival.json"
DEFAULT_CANDIDATES = ROOT / "reports" / "standard_causal_transformer_survival_candidates.jsonl"
DEFAULT_OUT = ROOT / "reports" / "standard_causal_transformer_survival_gate.json"
ALLOWED_READ_SET = {"prompt", "entry_point", "callable_signature"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=rel(DEFAULT_REPORT))
    parser.add_argument("--candidates", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    report_path = resolve(args.report)
    candidates_path = resolve(args.candidates)
    report = read_json(report_path)
    candidates = read_jsonl(candidates_path)
    gate = build_gate(report_path, candidates_path, report, candidates)
    write_json(resolve(args.out), gate)
    view = {
        "trigger_state": gate["trigger_state"],
        "summary": gate["summary"],
        "hard_gaps": gate["hard_gaps"],
        "adoption_gaps": gate["adoption_gaps"],
    }
    print(json.dumps(view if args.gate else gate, indent=2, sort_keys=True))
    return 2 if gate["trigger_state"] == "RED" else 0


def build_gate(
    report_path: Path,
    candidates_path: Path,
    report: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    adoption_gaps: list[dict[str, Any]] = []
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    stage = report.get("stage") if isinstance(report.get("stage"), dict) else {}
    training = report.get("training") if isinstance(report.get("training"), dict) else {}
    architecture = report.get("architecture") if isinstance(report.get("architecture"), dict) else {}
    checkpoint = resolve(str((report.get("artifacts") or {}).get("checkpoint") or ""))

    if not report_path.exists():
        hard_gaps.append(gap("report_missing", {"path": rel(report_path)}))
    if not candidates_path.exists():
        hard_gaps.append(gap("candidate_manifest_missing", {"path": rel(candidates_path)}))
    if report.get("policy") != "project_theseus_standard_causal_transformer_survival_v1":
        hard_gaps.append(gap("report_policy_mismatch", {"policy": report.get("policy")}))
    if architecture.get("family") != "standard_decoder_only_causal_transformer":
        hard_gaps.append(gap("architecture_family_mismatch", {"architecture": architecture}))
    if int(architecture.get("parameter_count") or 0) <= 0:
        hard_gaps.append(gap("parameter_count_missing", {"architecture": architecture}))
    if not checkpoint.exists() or checkpoint.stat().st_size <= 0:
        hard_gaps.append(gap("checkpoint_missing_or_empty", {"path": rel(checkpoint)}))
    if not training.get("complete"):
        adoption_gaps.append(gap("training_tranche_incomplete", {"training": training}, severity="adoption_gap"))
    if not training.get("eval_loss_improved"):
        adoption_gaps.append(gap("heldout_lm_loss_not_improved", {"training": training}, severity="adoption_gap"))

    for key in (
        "train_holdout_family_overlap_count",
        "train_eval_prompt_overlap_count",
        "train_eval_body_overlap_count",
        "licensed_pretrain_eval_body_overlap_source_surviving_count",
        "public_training_rows",
        "external_inference_calls",
    ):
        if int(stage.get(key) or 0) != 0:
            hard_gaps.append(gap(f"stage_{key}_nonzero", {"observed": stage.get(key)}))
    if int(stage.get("unique_semantic_eval_task_count") or 0) != int(
        stage.get("family_disjoint_eval_task_count") or 0
    ):
        hard_gaps.append(
            gap(
                "repeated_semantic_eval_tasks",
                {
                    "unique": stage.get("unique_semantic_eval_task_count"),
                    "reported": stage.get("family_disjoint_eval_task_count"),
                },
            )
        )
    if int(stage.get("unique_semantic_eval_task_count") or 0) < 24:
        hard_gaps.append(
            gap(
                "insufficient_distinct_eval_tasks",
                {"observed": stage.get("unique_semantic_eval_task_count"), "required": 24},
            )
        )
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int(report.get(key) or 0) != 0:
            hard_gaps.append(gap(f"report_{key}_nonzero", {"observed": report.get(key)}))

    parse_valid = 0
    task_ids: set[str] = set()
    duplicate_within_task: set[tuple[str, str]] = set()
    cross_task_duplicate_hashes: set[str] = set()
    observed_hash_tasks: dict[str, set[str]] = {}
    observed_body_hash_tasks: dict[str, set[str]] = {}
    family_counts: dict[str, int] = {}
    from candidate_integrity import recompute_candidate_integrity

    for index, row in enumerate(candidates):
        code = str(row.get("code") or "")
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if str(row.get("candidate_sha256") or "") != digest:
            hard_gaps.append(gap("candidate_hash_mismatch", {"index": index}))
        task_id = str(row.get("task_id") or "")
        prior_tasks = observed_hash_tasks.setdefault(digest, set())
        if task_id in prior_tasks:
            duplicate_within_task.add((task_id, digest))
        elif prior_tasks:
            cross_task_duplicate_hashes.add(digest)
        prior_tasks.add(task_id)
        try:
            parsed = ast.parse(code)
            parse_valid += 1
            body_digest = normalized_function_body_hash(parsed)
            body_tasks = observed_body_hash_tasks.setdefault(body_digest, set())
            if task_id not in body_tasks and body_tasks:
                cross_task_duplicate_hashes.add(body_digest)
            body_tasks.add(task_id)
        except SyntaxError as exc:
            hard_gaps.append(gap("candidate_syntax_invalid", {"index": index, "error": str(exc)}))
        if set(row.get("generation_read_set") or []) != ALLOWED_READ_SET:
            hard_gaps.append(
                gap(
                    "generation_read_set_mismatch",
                    {"index": index, "observed": row.get("generation_read_set"), "allowed": sorted(ALLOWED_READ_SET)},
                )
            )
        for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
            if int(row.get(key) or 0) != 0:
                hard_gaps.append(gap(f"candidate_{key}_nonzero", {"index": index, "observed": row.get(key)}))
        if row.get("fallback_return_used") is not False or row.get("body_template_selected") is not False:
            hard_gaps.append(gap("candidate_fallback_or_template_boundary_fault", {"index": index}))
        integrity = recompute_candidate_integrity(row)
        family = str(integrity.get("recomputed_candidate_family") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        if family not in {"transformer_hybrid", "learned_full_body_token"}:
            adoption_gaps.append(
                gap(
                    "candidate_not_independently_promotion_grade",
                    {"index": index, "integrity": integrity},
                    severity="adoption_gap",
                )
            )
        task_ids.add(task_id)

    if duplicate_within_task:
        hard_gaps.append(gap("duplicate_candidate_code_within_task", {"count": len(duplicate_within_task)}))
    if cross_task_duplicate_hashes:
        adoption_gaps.append(
            gap(
                "cross_task_candidate_collapse",
                {"duplicate_code_hash_count": len(cross_task_duplicate_hashes)},
                severity="adoption_gap",
            )
        )
    if int(summary.get("candidate_count") or 0) != len(candidates):
        hard_gaps.append(
            gap("candidate_count_mismatch", {"reported": summary.get("candidate_count"), "actual": len(candidates)})
        )
    if int(summary.get("syntax_valid_candidate_count") or 0) != parse_valid:
        hard_gaps.append(
            gap("syntax_count_mismatch", {"reported": summary.get("syntax_valid_candidate_count"), "actual": parse_valid})
        )
    passed = int(summary.get("model_only_passed_task_count") or 0)
    if passed <= 0:
        adoption_gaps.append(
            gap(
                "model_only_private_behavior_not_above_zero",
                {"passed": passed, "eval_tasks": summary.get("family_disjoint_eval_task_count")},
                severity="adoption_gap",
            )
        )
    adoption_state = "QUALIFIED" if not adoption_gaps and not hard_gaps else "NOT_ADOPTED"
    return {
        "policy": "project_theseus_standard_causal_transformer_survival_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_state": "GREEN" if not hard_gaps else "RED",
        "summary": {
            "candidate_count": len(candidates),
            "candidate_task_count": len(task_ids - {""}),
            "syntax_valid_candidate_count": parse_valid,
            "independent_family_counts": family_counts,
            "model_only_passed_task_count": passed,
            "hard_gap_count": len(hard_gaps),
            "adoption_gap_count": len(adoption_gaps),
            "adoption_state": adoption_state,
            "checkpoint": rel(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint) if checkpoint.exists() else "",
        },
        "hard_gaps": hard_gaps,
        "adoption_gaps": adoption_gaps,
        "rules": {
            "integrity": "candidate family and syntax are independently recomputed; candidate flags are not trusted",
            "adoption": "integrity can be GREEN for a valid negative, but route adoption requires complete training, improved heldout loss, and model-only family-disjoint behavior above zero",
            "no_credit": "templates, renderers, routers, tools, repairs, and fallback returns cannot satisfy this gate",
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def gap(kind: str, detail: dict[str, Any], *, severity: str = "hard_gap") -> dict[str, Any]:
    return {"kind": kind, "severity": severity, "detail": detail}


def normalized_function_body_hash(tree: ast.Module) -> str:
    function = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    body = function.body if function is not None else tree.body
    payload = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except ValueError:
        return str(value)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
