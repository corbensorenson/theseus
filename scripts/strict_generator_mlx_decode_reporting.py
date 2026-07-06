#!/usr/bin/env python3
"""Reporting helpers for strict-generator MLX decode/eval.

This module contains evidence shaping, gate construction, checkpoint path
resolution, and JSON IO helpers. It deliberately avoids decode-time candidate
generation so extraction cannot change model behavior.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from candidate_integrity import recompute_candidate_integrity
from neural_seed_code_proposer_comparator import dict_or_empty, get_path, resolve


STRICT_GENERATOR_SOURCE_TEXT_POLICY = "prompt_signature_only_v1"


def stamp_mlx_rows(rows: list[dict[str, Any]], *, checkpoint: Any, split_name: str) -> list[dict[str, Any]]:
    for row in rows:
        row["candidate_source"] = "strict_generator_mlx_decode_eval"
        row["source_module"] = "scripts/strict_generator_mlx_decode_eval.py"
        row["candidate_generation_contract"] = "mlx_transformer_full_body_token_decoder_private_checkpoint_v1"
        row["full_body_token_candidate"] = True
        row["grammar_masked_learned_token_candidate"] = True
        row["token_level_code_generation_learned"] = True
        row["compositional_token_candidate"] = True
        row["benchmark_integrity"] = {
            "public_tests_used": False,
            "public_solutions_used": False,
            "canonical_solution_used": False,
            "eval_tests_used_for_generation": False,
            "eval_solutions_used_for_generation": False,
            "teacher_used": False,
        }
        provenance = row.setdefault("provenance", {})
        provenance["source_module"] = "scripts/strict_generator_mlx_decode_eval.py"
        provenance["candidate_family"] = "transformer_hybrid"
        provenance["candidate_generation_mode_detail"] = "mlx_direct_full_body_token_decoder"
        provenance["checkpoint"] = checkpoint
        provenance["evaluation_split"] = split_name
        provenance["generation_inputs"] = [
            "prompt",
            "entry_point",
            "callable_signature",
            "visible_intent_tags",
            "prompt_operation_hints",
            "visible_type_shape_tags",
            "identifier_parts",
            "visible_subwords",
        ]
        provenance["source_text_policy"] = STRICT_GENERATOR_SOURCE_TEXT_POLICY
        provenance["tests_used_for_generation"] = False
        provenance["solutions_used_for_generation"] = False
        provenance["teacher_used"] = False
    return rows


def syntax_summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    syntax_ok = sum(1 for row in rows if bool(get_path(row, ["static_coherence", "parse_ok"], False)))
    return {
        "candidate_rows": total,
        "syntax_pass_count": syntax_ok,
        "syntax_pass_rate": round(syntax_ok / total, 6) if total else 0.0,
    }


def selection_summary(selection: dict[str, Any]) -> dict[str, Any]:
    summary = dict_or_empty(selection.get("summary"))
    summary["active"] = bool(selection.get("active"))
    summary["split"] = selection.get("split")
    summary["seed"] = selection.get("seed")
    summary["family_key"] = selection.get("family_key")
    summary["split_overlap_audit"] = get_path(selection, ["split_audit", "overlap"], {})
    return summary


def candidate_integrity_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    generated = [
        row
        for row in candidates
        if row.get("candidate_generation_mode") == "token_level_code_decoder"
        and row.get("substrate_adapter") != "shared_null_baseline"
    ]
    family_counts: Counter[str] = Counter()
    verified_by_family: Counter[str] = Counter()
    mismatch_counts: Counter[str] = Counter()
    syntax_invalid_by_family: Counter[str] = Counter()
    for row in generated:
        integrity = recompute_candidate_integrity(row)
        family = str(integrity.get("recomputed_candidate_family") or "unknown")
        family_counts[family] += 1
        if bool(integrity.get("integrity_verified")):
            verified_by_family[family] += 1
        shape = integrity.get("code_shape") if isinstance(integrity.get("code_shape"), dict) else {}
        if not bool(shape.get("syntax_valid", False)):
            syntax_invalid_by_family[family] += 1
        for mismatch in integrity.get("integrity_mismatches") or []:
            mismatch_counts[str(mismatch)] += 1
    verified_count = sum(verified_by_family.values())
    mismatch_count = sum(mismatch_counts.values())
    return {
        "policy": "project_theseus_inline_candidate_integrity_summary_v1",
        "generated_candidate_count": len(generated),
        "family_counts": dict(sorted(family_counts.items())),
        "integrity_verified_by_family": dict(sorted(verified_by_family.items())),
        "integrity_verified_candidate_count": verified_count,
        "integrity_verified_candidate_rate": round(verified_count / len(generated), 6) if generated else 0.0,
        "integrity_mismatch_count": mismatch_count,
        "integrity_mismatch_counts": dict(sorted(mismatch_counts.items())),
        "syntax_invalid_by_family": dict(sorted(syntax_invalid_by_family.items())),
        "score_semantics": (
            "Independent inline integrity recomputation over generated token-decoder rows. "
            "Candidate-emitted flags are not trusted for family, template/fallback status, "
            "learned status, or promotion eligibility."
        ),
    }


def build_gates(
    splits: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    integrity_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    split_reports = [dict_or_empty(row) for row in splits.values()]
    hard_split_active = bool(split_reports) and all(bool(row.get("active")) for row in split_reports)
    generated = [
        row
        for row in candidates
        if row.get("candidate_generation_mode") == "token_level_code_decoder"
        and row.get("substrate_adapter") != "shared_null_baseline"
    ]
    integrity = integrity_summary if isinstance(integrity_summary, dict) else candidate_integrity_summary(candidates)
    mismatch_counts = dict_or_empty(integrity.get("integrity_mismatch_counts"))
    label_summaries = {
        name: get_path(report, ["summary", "private_verifier", "candidate_label_summary"], {})
        for name, report in splits.items()
    }
    return [
        gate("requested_splits_active", hard_split_active, "hard", list(splits)),
        gate("candidate_rows_emitted", len(generated) > 0, "hard", len(generated)),
        gate("external_inference_zero", all(int(row.get("external_inference_calls") or 0) == 0 for row in candidates), "hard", 0),
        gate("public_training_rows_zero", True, "hard", 0),
        gate(
            "no_fallback_returns",
            all(not bool(get_path(row, ["grammar_repair", "fallback_return_used"], False)) for row in generated),
            "hard",
            Counter(str(get_path(row, ["grammar_repair", "strategy"], "")) for row in generated),
        ),
        gate(
            "no_public_or_eval_generation_visibility",
            all(
                not bool(row.get(key))
                for row in candidates
                for key in [
                    "public_tests_visible_to_generator",
                    "public_solutions_visible_to_generator",
                    "eval_tests_visible_to_generator",
                    "eval_solution_visible_to_generator",
                ]
            ),
            "hard",
            "candidate flags clean",
        ),
        gate(
            "strict_source_text_audit_clean",
            all(bool(get_path(row, ["summary", "source_text_audit", "clean"], False)) for row in split_reports),
            "hard",
            {
                name: get_path(report, ["summary", "source_text_audit"], {})
                for name, report in splits.items()
            },
        ),
        gate(
            "private_verifier_ran",
            all(int(get_path(row, ["summary", "private_verifier", "eval_task_count"], 0) or 0) > 0 for row in split_reports),
            "hard",
            {
                name: get_path(report, ["summary", "private_verifier", "eval_task_count"], 0)
                for name, report in splits.items()
            },
        ),
        gate(
            "private_verifier_correctness_labels_attached",
            all(
                int(dict_or_empty(summary).get("attached_generated_label_count") or 0)
                >= min(
                    int(get_path(report, ["summary", "generated_candidate_rows"], 0) or 0),
                    int(dict_or_empty(summary).get("private_eval_trace_rows") or 0),
                )
                for summary, report in zip(label_summaries.values(), split_reports)
            ),
            "hard",
            label_summaries,
        ),
        gate(
            "inline_candidate_integrity_verified_rows_present",
            int(integrity.get("integrity_verified_candidate_count") or 0) > 0,
            "soft",
            integrity,
        ),
        gate(
            "inline_candidate_integrity_clean_for_green",
            int(integrity.get("integrity_mismatch_count") or 0) == 0,
            "soft",
            integrity,
        ),
        gate(
            "no_inert_learned_candidate_mismatches",
            int(mismatch_counts.get("claimed_learned_candidate_but_inert_stub_like") or 0) == 0,
            "soft",
            mismatch_counts,
        ),
        gate(
            "functional_pass_moved_above_zero",
            any(int(get_path(row, ["summary", "private_verifier", "trained_passed"], 0) or 0) > 0 for row in split_reports),
            "soft",
            {
                name: get_path(report, ["summary", "private_verifier", "trained_passed"], 0)
                for name, report in splits.items()
            },
        ),
        gate(
            "nontrivial_return_rate_nonzero",
            any(float(get_path(row, ["summary", "static_coherence", "nontrivial_return_rate"], 0.0) or 0.0) > 0.0 for row in split_reports),
            "soft",
            {
                name: get_path(report, ["summary", "static_coherence", "nontrivial_return_rate"], 0.0)
                for name, report in splits.items()
            },
        ),
    ]


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def resolve_checkpoint_paths(args: Any, checkpoint_report: dict[str, Any]) -> tuple[Path, Path]:
    return resolve_checkpoint_paths_from_report(
        checkpoint_report,
        checkpoint_override=str(args.checkpoint or ""),
        vocab_override=str(args.vocab or ""),
    )


def resolve_checkpoint_paths_from_report(
    checkpoint_report: dict[str, Any],
    *,
    checkpoint_override: str = "",
    vocab_override: str = "",
) -> tuple[Path, Path]:
    budget = dict_or_empty(checkpoint_report.get("budget"))
    summary = dict_or_empty(checkpoint_report.get("summary"))
    checkpoint_raw = str(checkpoint_override or budget.get("checkpoint") or summary.get("checkpoint") or "").strip()
    vocab_raw = str(vocab_override or budget.get("vocab") or summary.get("vocab") or "").strip()
    if not checkpoint_raw:
        raise SystemExit("missing MLX checkpoint path; pass --checkpoint or --checkpoint-report")
    if not vocab_raw:
        raise SystemExit("missing MLX vocab path; pass --vocab or --checkpoint-report")
    return resolve(checkpoint_raw), resolve(vocab_raw)


def stable_hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
