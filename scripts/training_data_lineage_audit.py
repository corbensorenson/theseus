#!/usr/bin/env python3
"""Candidate-level data admission, lineage, and lifecycle governance.

The ledger stores hashes and policy receipts, never training payload text. Public
benchmark material is read only to build an in-memory contamination index and is
never copied into a receipt or training artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import full_state_update_causality


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADMISSION = ROOT / "reports" / "training_data_admission_v1.json"
DEFAULT_TEACHER_MANIFEST = ROOT / "reports" / "teacher_distillation_manifest.json"
DEFAULT_LEDGER = ROOT / "runtime" / "data_governance" / "data_admission_receipts_v1.jsonl.gz"
DEFAULT_OUT = ROOT / "reports" / "training_data_lineage_audit.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "training_data_lineage_audit.md"
DEFAULT_FULL_STATE_CONTRACT = ROOT / "configs" / "full_state_update_causality.json"

POLICY = "project_theseus_candidate_data_lineage_governance_v2"
TEXT_KEYS = {
    "prompt", "instruction", "question", "solution", "solution_body", "canonical_solution",
    "code", "answer", "tests", "input", "output", "completion", "response",
}
PUBLIC_TRUE_KEYS = {
    "public_benchmark", "public_benchmark_row", "public_benchmark_payload",
    "public_prompt", "public_solution", "public_solutions_used", "public_tests_used",
    "public_prompts_included", "public_score_labels_included", "uses_public_data",
}
RAW_USER_KEYS = {"raw_user_text", "raw_prompt", "conversation_text", "private_user_text"}
ALLOWED_LICENSES = {
    "cc0-1.0", "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc",
    "project-internal", "project-internal-private", "private-generated",
}
FALLBACK_MARKERS = {
    "fallback_return", "expression_memory_fallback", "placeholder_scaffold_body",
    "return none  # fallback", "return 0  # fallback",
}
MINHASH_SEEDS = (
    0x9E3779B185EBCA87,
    0xC2B2AE3D27D4EB4F,
    0x165667B19E3779F9,
    0x85EBCA77C2B2AE63,
    0x27D4EB2F165667C5,
    0x94D049BB133111EB,
    0xD6E8FEB86659FD93,
    0xA0761D6478BD642F,
)
MASK64 = (1 << 64) - 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", default=rel(DEFAULT_ADMISSION))
    parser.add_argument("--teacher-manifest", default=rel(DEFAULT_TEACHER_MANIFEST))
    parser.add_argument("--ledger-out", default=rel(DEFAULT_LEDGER))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-public-texts", type=int, default=20000)
    args = parser.parse_args()

    started = time.perf_counter()
    admission = read_json(resolve(args.admission))
    bundle = build_lineage_bundle(
        admission,
        teacher_manifest_path=resolve(args.teacher_manifest),
        ledger_path=resolve(args.ledger_out),
        max_rows=max(0, args.max_rows),
        max_public_texts=max(100, args.max_public_texts),
        started=started,
    )
    write_json(resolve(args.out), bundle)
    write_text(resolve(args.markdown_out), render_markdown(bundle))
    print(json.dumps(compact_console_view(bundle), indent=2, sort_keys=True))
    return 0 if bundle["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_lineage_bundle(
    admission: dict[str, Any],
    *,
    teacher_manifest_path: Path = DEFAULT_TEACHER_MANIFEST,
    ledger_path: Path = DEFAULT_LEDGER,
    max_rows: int = 0,
    max_public_texts: int = 20000,
    started: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter() if started is None else started
    contamination_index = build_public_contamination_index(max_texts=max_public_texts)
    source_rows = [
        row for row in as_list(admission.get("source_admissions"))
        if isinstance(row, dict) and row.get("allowed_for_training") is True
    ]
    teacher_manifest = read_json(teacher_manifest_path)
    input_digest = stable_id(
        "candidate-lineage-input",
        POLICY,
        file_sha256(Path(__file__)),
        [
            {
                "source_id": row.get("source_id"),
                "sha256": row.get("sha256"),
                "row_count": row.get("row_count"),
                "license_status": row.get("license_status"),
                "provenance_status": row.get("provenance_status"),
            }
            for row in source_rows
        ],
        file_sha256(teacher_manifest_path),
        contamination_index["digest"],
        file_sha256(Path(full_state_update_causality.__file__)),
        file_sha256(DEFAULT_FULL_STATE_CONTRACT),
        max_rows,
        max_public_texts,
        rel(ledger_path),
    )
    expected_count = min(
        max_rows or sum(int(row.get("row_count") or 0) for row in source_rows) + len(as_list(teacher_manifest.get("rows"))),
        sum(int(row.get("row_count") or 0) for row in source_rows) + len(as_list(teacher_manifest.get("rows"))),
    )
    migrated_teacher_receipts = upgrade_teacher_lineage_receipts(
        ledger_path,
        teacher_manifest=teacher_manifest,
        teacher_manifest_path=teacher_manifest_path,
        contamination_index=contamination_index,
    )
    cached = reusable_bundle(
        input_digest=input_digest,
        ledger_path=ledger_path,
        started=started,
        migrated_teacher_receipts=migrated_teacher_receipts,
    )
    if cached:
        return cached
    metrics: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    source_receipt_counts: Counter[str] = Counter()
    workload: list[dict[str, Any]] = []
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    recovered = recover_ledger_state(
        ledger_path,
        source_rows=source_rows,
        teacher_manifest_path=teacher_manifest_path,
        contamination_digest=contamination_index["digest"],
        expected_count=expected_count,
    )
    incremental = {} if recovered or max_rows else incremental_rebuild_ledger(
        ledger_path,
        source_rows=source_rows,
        teacher_manifest=teacher_manifest,
        teacher_manifest_path=teacher_manifest_path,
        contamination_index=contamination_index,
        expected_count=expected_count,
    )
    replay_state = recovered or incremental
    if replay_state:
        processed = replay_state["processed"]
        metrics.update(replay_state["metrics"])
        class_counts.update(replay_state["class_counts"])
        depth_counts.update(replay_state["depth_counts"])
        family_counts.update(replay_state["family_counts"])
        source_receipt_counts.update(replay_state["source_receipt_counts"])
        workload.extend(replay_state["workload"])
    else:
        temporary = tempfile.NamedTemporaryFile(prefix="data-admission-", suffix=".jsonl.gz", dir=ledger_path.parent, delete=False)
        temporary_path = Path(temporary.name)
        temporary.close()
        processed = 0
        try:
            with gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=6) as handle:
                for source in source_rows:
                    source_path = resolve(str(source.get("path") or ""))
                    for row_index, row in enumerate(iter_rows(source_path)):
                        if max_rows and processed >= max_rows:
                            break
                        receipt = candidate_receipt(
                            row,
                            source=source,
                            row_index=row_index,
                            contamination_index=contamination_index,
                        )
                        handle.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
                        processed += 1
                        update_metrics(metrics, class_counts, depth_counts, family_counts, source_receipt_counts, receipt)
                        workload.append(workload_item(receipt, processed))
                    if max_rows and processed >= max_rows:
                        break

                for row_index, row in enumerate(as_list(teacher_manifest.get("rows"))):
                    if max_rows and processed >= max_rows:
                        break
                    if not isinstance(row, dict):
                        continue
                    receipt = teacher_candidate_receipt(
                        row,
                        row_index=row_index,
                        manifest=teacher_manifest,
                        manifest_path=teacher_manifest_path,
                        contamination_index=contamination_index,
                    )
                    handle.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
                    processed += 1
                    update_metrics(metrics, class_counts, depth_counts, family_counts, source_receipt_counts, receipt)
                    workload.append(workload_item(receipt, processed))
            os.replace(temporary_path, ledger_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    ledger_receipt = replay_state.get("ledger_receipt") if replay_state else audit_ledger(ledger_path)
    adversary = run_adversary_controls()
    continual = continual_learning_comparison(workload)
    deletion = descendant_deletion_closure_fixture()
    full_state = full_state_update_causality.run_reference_fixture(
        full_state_update_causality.load_contract(DEFAULT_FULL_STATE_CONTRACT)
    )
    recursive = recursive_synthetic_diagnostics(
        metrics=metrics,
        class_counts=class_counts,
        depth_counts=depth_counts,
        family_counts=family_counts,
    )
    hard_gaps = []
    if admission.get("trigger_state") not in {"GREEN", "YELLOW"}:
        hard_gaps.append({"kind": "source_admission_not_ready", "state": admission.get("trigger_state")})
    if ledger_receipt["receipt_count"] != processed or not ledger_receipt["replay_valid"]:
        hard_gaps.append({"kind": "candidate_receipt_ledger_replay_failed", "ledger": ledger_receipt})
    if adversary["passed_count"] != adversary["case_count"]:
        hard_gaps.append({"kind": "provenance_or_contamination_adversary_failed", "controls": adversary})
    if continual["policy_count"] != 5 or not continual["comparison_ready"]:
        hard_gaps.append({"kind": "continual_policy_comparison_incomplete", "comparison": continual})
    if not deletion["positive_fixture_closed"] or not deletion["expected_invalid_fixture_rejected"]:
        hard_gaps.append({"kind": "descendant_deletion_closure_fixture_failed", "closure": deletion})
    if full_state.get("trigger_state") != "GREEN":
        hard_gaps.append({"kind": "full_state_update_causality_failed", "fixture": full_state})
    if int(metrics.get("admitted_exact_public_overlap") or 0) or int(metrics.get("admitted_semantic_public_overlap") or 0):
        hard_gaps.append({
            "kind": "contaminated_candidate_admitted",
            "exact": int(metrics.get("admitted_exact_public_overlap") or 0),
            "semantic": int(metrics.get("admitted_semantic_public_overlap") or 0),
        })

    warnings = []
    if int(metrics.get("quarantine") or 0):
        warnings.append({"kind": "candidate_receipts_quarantined", "count": int(metrics["quarantine"])})
    if recursive["unknown_lineage_depth_count"]:
        warnings.append({"kind": "synthetic_lineage_depth_unknown", "count": recursive["unknown_lineage_depth_count"]})
    if recursive["collapse_risk_flags"]:
        warnings.append({
            "kind": "recursive_synthetic_distribution_risk",
            "flags": recursive["collapse_risk_flags"],
            "synthetic_share": recursive["synthetic_share"],
            "non_claim": "This distribution warning is not a model-collapse diagnosis.",
        })
    trigger_state = "RED" if hard_gaps else ("YELLOW" if warnings else "GREEN")
    summary = {
        "candidate_receipt_count": processed,
        "admitted_candidate_count": int(metrics.get("admit") or 0),
        "quarantined_candidate_count": int(metrics.get("quarantine") or 0),
        "rejected_candidate_count": int(metrics.get("reject") or 0),
        "source_count": len(source_rows),
        "teacher_candidate_count": int(class_counts.get("teacher_distillation") or 0),
        "exact_public_overlap_candidate_count": int(metrics.get("exact_overlap") or 0),
        "semantic_public_overlap_candidate_count": int(metrics.get("semantic_overlap") or 0),
        "public_flag_candidate_count": int(metrics.get("public_flag") or 0),
        "fallback_marker_candidate_count": int(metrics.get("fallback_marker") or 0),
        "raw_user_text_candidate_count": int(metrics.get("raw_user") or 0),
        "decision_reason_counts": {
            key.removeprefix("reason:"): value
            for key, value in sorted(metrics.items())
            if key.startswith("reason:")
        },
        "lineage_edge_count": int(metrics.get("lineage_edges") or 0),
        "family_count": len(family_counts),
        "admitted_hash_filter_ready": bool(processed and metrics.get("admit")),
        "adversary_case_count": adversary["case_count"],
        "adversary_passed_count": adversary["passed_count"],
        "continual_policy_count": continual["policy_count"],
        "deletion_artifact_kind_count": deletion["artifact_kind_count"],
        "full_state_causality_state": full_state.get("trigger_state"),
        "full_state_artifact_kind_count": (full_state.get("summary") or {}).get("artifact_kind_count", 0),
        "full_state_mutation_passed_count": (full_state.get("summary") or {}).get("mutation_passed_count", 0),
        "full_state_mutation_case_count": (full_state.get("summary") or {}).get("mutation_case_count", 0),
        "full_state_exact_rollback": bool((full_state.get("rollback") or {}).get("exact_pre_state_restored")),
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
        "public_training_rows_written": 0,
        "runtime_external_inference_calls": 0,
        "fallback_return_count": 0,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "report_cache_reused": False,
        "ledger_replay_reused": bool(recovered),
        "ledger_incremental_rebuild_used": bool(incremental),
        "ledger_incremental_reused_source_count": int(incremental.get("reused_source_count") or 0),
        "ledger_incremental_rescanned_source_count": int(incremental.get("rescanned_source_count") or 0),
        "ledger_incremental_reused_receipt_count": int(incremental.get("reused_receipt_count") or 0),
        "ledger_incremental_rescanned_receipt_count": int(incremental.get("rescanned_receipt_count") or 0),
        "ledger_schema_migrated_teacher_receipt_count": migrated_teacher_receipts,
    }
    records = build_viea_records(summary, ledger_receipt, continual, deletion, adversary, full_state)
    return {
        "policy": POLICY,
        "created_utc": now(),
        "input_digest": input_digest,
        "trigger_state": trigger_state,
        "summary": summary,
        "candidate_receipt_ledger": ledger_receipt,
        "source_receipt_counts": dict(sorted(source_receipt_counts.items())),
        "provenance_class_counts": dict(sorted(class_counts.items())),
        "recursive_synthetic_diagnostics": recursive,
        "semantic_and_provenance_adversary_controls": adversary,
        "continual_learning_policy_comparison": continual,
        "descendant_deletion_closure": deletion,
        "full_state_update_causality": full_state,
        "viea_data_governance_records": records,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "score_semantics": (
            "Candidate-level admission, contamination, lineage, continual-policy simulation, full-state update/rollback, and deletion-closure evidence only. "
            "It does not train a model, prove unlearning, prove model quality, or spend public calibration."
        ),
        "non_claims": [
            "Receipt completeness is not data quality or model capability.",
            "The continual-learning workload is a frozen policy simulation, not a training result.",
            "Deletion fixtures prove propagation logic, not erasure from every deployed external system.",
            "Full-state package replay proves bounded mechanics, not behavioral unlearning or model improvement.",
            "Semantic overlap heuristics can quarantine candidates but cannot prove absence of all contamination.",
            "Teacher proposals are not served directly and do not become training rows without the teacher gate.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def candidate_receipt(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
    row_index: int,
    contamination_index: dict[str, Any],
) -> dict[str, Any]:
    row_hash = row_sha256(row)
    source_id = str(source.get("source_id") or stable_id("source", source.get("path")))
    source_path = str(source.get("path") or "")
    flat = flatten(row)
    texts = extract_texts(row)
    exact_count = sum(1 for text in texts if text_sha256(text) in contamination_index["exact_hashes"])
    semantic_segment_count, semantic_max = semantic_overlap(texts, contamination_index)
    semantic_count = int(semantic_segment_count >= 2 or semantic_max >= 0.92)
    public_flag = any(truthy(flat.get(key)) for key in PUBLIC_TRUE_KEYS)
    raw_user = any(bool(flat.get(key)) for key in RAW_USER_KEYS)
    fallback_marker = any(marker in json.dumps(row, sort_keys=True).lower() for marker in FALLBACK_MARKERS)
    split = str(row.get("split") or "train").lower()
    heldout = split in {"eval", "test", "heldout", "validation"}
    row_license = normalize_license(first_value(flat, ("license_spdx", "license", "spdx")))
    source_license = str(source.get("license_status") or "")
    license_allowed = row_license in ALLOWED_LICENSES or source_license.startswith(("allowed_", "project_internal_"))
    provenance_class, synthetic_depth, parent_refs = provenance_profile(row, source)
    verifier_accepted = truthy(flat.get("accepted")) or str(source.get("source_kind")) != "teacher_distillation"
    reasons = []
    if public_flag:
        reasons.append("public_benchmark_flag_true")
    if exact_count:
        reasons.append("exact_public_overlap")
    if semantic_count:
        reasons.append("semantic_public_overlap")
    if raw_user:
        reasons.append("raw_user_text_present")
    if fallback_marker:
        reasons.append("fallback_marker_present")
    if heldout:
        reasons.append("heldout_split")
    if not license_allowed:
        reasons.append("license_not_allowed")
    if not verifier_accepted:
        reasons.append("verifier_not_accepted")
    if provenance_class == "unknown":
        reasons.append("lineage_incomplete")
    decision = "admit" if not reasons else (
        "quarantine" if any("overlap" in reason or reason == "lineage_incomplete" for reason in reasons) else "reject"
    )
    candidate_id = stable_id("candidate", source_id, row_index, row_hash)
    family = family_label(row)
    return {
        "record_type": "data_admission_receipt",
        "policy": POLICY,
        "receipt_id": stable_id("data-admission", candidate_id, decision),
        "candidate_id": candidate_id,
        "row_sha256": row_hash,
        "row_index": row_index,
        "source_id": source_id,
        "source_path": source_path,
        "source_sha256": source.get("sha256"),
        "source_kind": source.get("source_kind"),
        "family": family,
        "authority": {
            "decision_source": "training_data_admission_v1_source_receipt_plus_candidate_audit",
            "training_allowed": decision == "admit",
            "runtime_direct_serving_allowed": False,
            "public_calibration_payload_use_allowed": False,
        },
        "license_receipt": {
            "spdx": row_license or "source-level",
            "source_license_status": source_license,
            "allowed": license_allowed,
        },
        "provenance": {
            "class": provenance_class,
            "synthetic_depth": synthetic_depth,
            "parent_ref_hashes": parent_refs,
            "lineage_complete": provenance_class != "unknown" and (synthetic_depth == 0 or bool(parent_refs)),
        },
        "permitted_uses": ["private_student_training", "private_verifier_training", "audit_replay"],
        "forbidden_uses": ["runtime_direct_token_serving", "public_benchmark_training", "hidden_test_exposure"],
        "split": split,
        "split_exclusions": ["public_calibration", "family_disjoint_eval", "raw_user_text", "runtime_external_inference"],
        "contamination": {
            "public_index_digest": contamination_index["digest"],
            "exact_overlap_count": exact_count,
            "semantic_overlap_count": semantic_count,
            "semantic_segment_match_count": semantic_segment_count,
            "semantic_max_jaccard": round(semantic_max, 6),
            "heuristic_non_exhaustive": True,
        },
        "retention": {
            "class": "private_training_receipt",
            "revocation_subjects": [candidate_id, source_id, row_hash],
            "deletion_scope": "source_row_and_all_registered_descendants",
            "unverified_descendants_must_remain_visible": True,
        },
        "evaluation_refs": evaluation_ref_hashes(row),
        "residuals": [family],
        "decision": decision,
        "decision_reasons": reasons,
        "public_flag_detected": public_flag,
        "raw_user_text_detected": raw_user,
        "fallback_marker_detected": fallback_marker,
        "verifier_accepted": verifier_accepted,
        "lineage_edges": [
            {"parent": source_id, "child": candidate_id, "relation": "contains_row"},
            *({"parent": parent, "child": candidate_id, "relation": "derived_from"} for parent in parent_refs),
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_payload_stored": False,
    }


def teacher_candidate_receipt(
    row: dict[str, Any],
    *,
    row_index: int,
    manifest: dict[str, Any],
    manifest_path: Path,
    contamination_index: dict[str, Any],
) -> dict[str, Any]:
    training_row = as_dict(row.get("training_row"))
    source = {
        "source_id": stable_id("teacher-manifest", file_sha256(manifest_path)),
        "path": rel(manifest_path),
        "sha256": file_sha256(manifest_path),
        "source_kind": "teacher_distillation",
        "license_status": "allowed_" + normalize_license(row.get("license_spdx") or "project-internal"),
    }
    receipt = candidate_receipt(
        training_row,
        source=source,
        row_index=row_index,
        contamination_index=contamination_index,
    )
    checks = as_dict(row.get("admission_checks"))
    verifier = as_dict(row.get("local_verifier"))
    teacher_ready = bool(
        verifier.get("accepted") is True
        and all(checks.get(key) is True for key in (
            "leakage_audited", "license_checked", "provenance_retained",
            "public_benchmark_excluded", "runtime_serving_forbidden", "verifier_accepted",
        ))
        and manifest.get("admission_safety_checks_clean") is True
        and int(manifest.get("public_overlap_hits") or 0) == 0
    )
    reasons = [reason for reason in receipt["decision_reasons"] if reason != "verifier_not_accepted"]
    if not teacher_ready:
        reasons.append("teacher_distillation_gate_not_satisfied")
    receipt["decision_reasons"] = sorted(set(reasons))
    receipt["decision"] = "admit" if not reasons else "reject"
    receipt["authority"]["training_allowed"] = receipt["decision"] == "admit"
    receipt["authority"]["runtime_direct_serving_allowed"] = False
    receipt["provenance"]["class"] = "teacher_distillation"
    manifest_ref = stable_id("teacher-manifest", file_sha256(manifest_path))
    receipt["provenance"]["synthetic_depth"] = 1
    receipt["provenance"]["parent_ref_hashes"] = sorted(set([
        *as_list(receipt["provenance"].get("parent_ref_hashes")),
        manifest_ref,
    ]))
    receipt["provenance"]["lineage_complete"] = True
    receipt["provenance"]["teacher_request_id_hash"] = stable_id("teacher-request", row.get("request_id"))
    receipt["provenance"]["teacher_generation_calls_recorded"] = int(row.get("external_inference_calls") or 0)
    receipt["verifier_accepted"] = teacher_ready
    receipt["receipt_id"] = stable_id("teacher-data-admission", receipt["candidate_id"], receipt["decision"])
    receipt["lineage_edges"] = [
        *as_list(receipt.get("lineage_edges")),
        {"parent": manifest_ref, "child": receipt["candidate_id"], "relation": "governed_teacher_manifest"},
    ]
    return receipt


def build_public_contamination_index(*, max_texts: int) -> dict[str, Any]:
    exact_hashes: set[str] = set()
    entries: list[frozenset[int]] = []
    buckets: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    public_roots = [
        ROOT / "data" / "public_code_benchmark_manifests",
        ROOT / "data" / "old_project_benchmarks" / "cases",
        ROOT / "data" / "public_benchmarks" / "vcm_memory_quarantine",
    ]
    for root in public_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
                continue
            for row in iter_rows(path):
                for text in extract_texts(row):
                    normalized = normalize_text(text)
                    if len(normalized) < 8:
                        continue
                    exact_hashes.add(text_sha256(normalized))
                    tokens = token_hashes(normalized)
                    if len(tokens) >= 6:
                        entry_id = len(entries)
                        entries.append(tokens)
                        signature = minhash(tokens)
                        for band in minhash_bands(signature):
                            buckets[band].add(entry_id)
                    if len(exact_hashes) >= max_texts:
                        break
                if len(exact_hashes) >= max_texts:
                    break
            if len(exact_hashes) >= max_texts:
                break
        if len(exact_hashes) >= max_texts:
            break
    digest = hashlib.sha256("\n".join(sorted(exact_hashes)).encode("utf-8")).hexdigest()
    return {
        "exact_hashes": exact_hashes,
        "entries": entries,
        "buckets": buckets,
        "digest": digest,
        "text_count": len(exact_hashes),
        "semantic_entry_count": len(entries),
    }


def semantic_overlap(texts: list[str], index: dict[str, Any], *, threshold: float = 0.70) -> tuple[int, float]:
    match_count = 0
    best = 0.0
    entries = index["entries"]
    buckets = index["buckets"]
    for text in texts:
        tokens = token_hashes(normalize_text(text))
        if len(tokens) < 6:
            continue
        candidates: set[int] = set()
        for band in minhash_bands(minhash(tokens)):
            candidates.update(buckets.get(band, set()))
        local_best = 0.0
        for entry_id in candidates:
            other = entries[entry_id]
            union = len(tokens | other)
            score = len(tokens & other) / union if union else 0.0
            local_best = max(local_best, score)
        best = max(best, local_best)
        if local_best >= threshold:
            match_count += 1
    return match_count, best


def run_adversary_controls() -> dict[str, Any]:
    fixture_public = {
        "prompt": "Compute a stable weighted total from signed integer records while skipping invalid labels and preserving deterministic order.",
        "solution_body": "total = 0\nfor record in records:\n    if record.valid:\n        total += record.weight * abs(record.value)\nreturn total",
    }
    index = contamination_index_from_rows([fixture_public])
    source = {
        "source_id": "fixture-private-source",
        "path": "fixture://private",
        "sha256": stable_id("fixture-source"),
        "source_kind": "private_training_rows",
        "license_status": "allowed_cc0-1.0",
        "provenance_status": "project_internal_path_provenance",
    }
    cases = [
        ("valid_private", {"prompt": "Return the median timestamp from a validated local event stream.", "solution_body": "values = sorted(events)\nreturn values[len(values) // 2]", "license_spdx": "CC0-1.0", "split": "train"}, "admit"),
        ("exact_overlap", fixture_public, "quarantine"),
        ("semantic_overlap", {"prompt": "Compute a stable weighted total from signed integer entries while ignoring invalid labels and keeping deterministic order.", "solution_body": "result = 0\nfor item in records:\n    if item.valid:\n        result += item.weight * abs(item.value)\nreturn result", "license_spdx": "CC0-1.0"}, "quarantine"),
        ("public_flag", {"prompt": "private fixture", "solution_body": "return value", "public_tests_used": True, "license_spdx": "CC0-1.0"}, "reject"),
        ("missing_license", {"prompt": "private fixture", "solution_body": "return value"}, "reject"),
        ("teacher_unverified", {"prompt": "private fixture", "solution_body": "return value", "license_spdx": "project-internal"}, "reject"),
        ("raw_user", {"prompt": "private fixture", "solution_body": "return value", "raw_user_text": "present", "license_spdx": "CC0-1.0"}, "reject"),
        ("fallback", {"prompt": "private fixture", "solution_body": "return 0  # fallback", "license_spdx": "CC0-1.0"}, "reject"),
        ("heldout", {"prompt": "private fixture", "solution_body": "return value", "license_spdx": "CC0-1.0", "split": "test"}, "reject"),
    ]
    results = []
    for index_value, (case_id, row, expected) in enumerate(cases):
        case_source = dict(source)
        if case_id == "missing_license":
            case_source["license_status"] = "blocked_missing_license"
        if case_id == "teacher_unverified":
            case_source["source_kind"] = "teacher_distillation"
        receipt = candidate_receipt(row, source=case_source, row_index=index_value, contamination_index=index)
        observed = receipt["decision"]
        results.append({
            "case_id": case_id,
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
            "reason_codes": receipt["decision_reasons"],
        })
    return {
        "case_count": len(results),
        "passed_count": sum(1 for row in results if row["passed"]),
        "results": results,
        "raw_fixture_text_emitted": False,
    }


def contamination_index_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_hashes: set[str] = set()
    entries: list[frozenset[int]] = []
    buckets: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for row in rows:
        for text in extract_texts(row):
            exact_hashes.add(text_sha256(text))
            tokens = token_hashes(text)
            if len(tokens) < 6:
                continue
            entry_id = len(entries)
            entries.append(tokens)
            for band in minhash_bands(minhash(tokens)):
                buckets[band].add(entry_id)
    return {
        "exact_hashes": exact_hashes,
        "entries": entries,
        "buckets": buckets,
        "digest": hashlib.sha256("\n".join(sorted(exact_hashes)).encode()).hexdigest(),
    }


def continual_learning_comparison(workload: list[dict[str, Any]]) -> dict[str, Any]:
    admitted = [row for row in workload if row["decision"] == "admit"]
    if not admitted:
        return {"comparison_ready": False, "policy_count": 0, "policies": []}
    family_counts = Counter(row["family"] for row in admitted)
    all_families = set(family_counts)
    tail_limit = max(2, int(math.sqrt(len(admitted) / max(1, len(all_families)))))
    tail_families = {family for family, count in family_counts.items() if count <= tail_limit}
    cutoff = max(1, int(len(admitted) * 0.75))
    policies = {
        "replacement": [row for row in admitted if row["order"] >= cutoff and not row["simulated_revoked"]],
        "accumulation": list(admitted),
        "targeted_replay": [
            row for row in admitted
            if not row["simulated_revoked"]
            and (row["order"] >= cutoff or row["family"] in tail_families or stable_mod(row["candidate_id"], 5) == 0)
        ],
        "quarantine": [
            row for row in admitted
            if not row["simulated_revoked"] and row["lineage_complete"] and row["synthetic_depth"] <= 2
        ],
        "full_retraining": [row for row in admitted if not row["simulated_revoked"]],
    }
    multipliers = {"replacement": 1.0, "accumulation": 1.0, "targeted_replay": 1.2, "quarantine": 1.05, "full_retraining": 1.5}
    policy_rows = []
    old_families = {row["family"] for row in admitted if row["order"] < cutoff}
    for policy_id, selected in policies.items():
        selected_families = {row["family"] for row in selected}
        selected_tail = selected_families & tail_families
        coverage = len(selected_families) / max(1, len(all_families))
        tail_retention = len(selected_tail) / max(1, len(tail_families))
        forgetting = 1.0 - len(selected_families & old_families) / max(1, len(old_families))
        freshness = sum(row["order"] / max(1, len(admitted)) for row in selected) / max(1, len(selected))
        calibration = sum(1.0 if row["lineage_complete"] else 0.0 for row in selected) / max(1, len(selected))
        privacy_risk = sum(1 for row in selected if row["simulated_revoked"]) / max(1, len(selected))
        utility = 0.30 * coverage + 0.25 * tail_retention + 0.20 * (1.0 - forgetting) + 0.15 * freshness + 0.10 * calibration - 0.30 * privacy_risk
        policy_rows.append({
            "policy_id": policy_id,
            "selected_candidate_count": len(selected),
            "utility_proxy": round(utility, 6),
            "forgetting_proxy": round(forgetting, 6),
            "family_coverage": round(coverage, 6),
            "tail_retention": round(tail_retention, 6),
            "calibration_proxy": round(calibration, 6),
            "freshness": round(freshness, 6),
            "privacy_revocation_risk": round(privacy_risk, 6),
            "storage_units": len(selected),
            "compute_units": round(len(selected) * multipliers[policy_id], 3),
            "deletion_cost_units": sum(1 for row in selected if row["simulated_revoked"]),
        })
    eligible = [row for row in policy_rows if row["privacy_revocation_risk"] == 0.0]
    recommended = max(eligible, key=lambda row: (row["utility_proxy"], -row["compute_units"])) if eligible else None
    workload_hash = stable_id("continual-workload", [row["candidate_id"] for row in admitted])
    return {
        "policy": "project_theseus_frozen_continual_learning_policy_workload_v1",
        "comparison_ready": len(policy_rows) == 5,
        "policy_count": len(policy_rows),
        "workload_candidate_count": len(admitted),
        "workload_hash": workload_hash,
        "tail_family_count": len(tail_families),
        "policies": policy_rows,
        "recommended_policy": recommended["policy_id"] if recommended else "none",
        "score_semantics": "Metadata-policy simulation only; utility/forgetting/calibration are deterministic proxies, not trained-model measurements.",
    }


def descendant_deletion_closure_fixture() -> dict[str, Any]:
    kinds = [
        "source_row", "transform", "dataset", "checkpoint", "adapter", "cache",
        "vcm_index", "retrieval_index", "distilled_artifact", "report", "publication",
    ]
    nodes = [{"id": f"node-{kind}", "kind": kind} for kind in kinds]
    edges = [(nodes[index]["id"], nodes[index + 1]["id"]) for index in range(len(nodes) - 1)]
    positive = propagate_deletion(nodes, edges, roots={nodes[0]["id"]})
    invalid_edges = [edge for edge in edges if edge != ("node-checkpoint", "node-adapter")]
    negative = propagate_deletion(nodes, invalid_edges, roots={nodes[0]["id"]})
    return {
        "policy": "project_theseus_descendant_deletion_closure_v1",
        "artifact_kinds": kinds,
        "artifact_kind_count": len(kinds),
        "positive_fixture_closed": positive["closed"],
        "positive_fixture": positive,
        "expected_invalid_fixture_rejected": not negative["closed"] and bool(negative["unverified_descendants"]),
        "expected_invalid_fixture": negative,
        "real_deletion_executed": False,
        "non_claim": "The fixture proves graph propagation and unverified-descendant retention, not physical erasure from unregistered external systems.",
    }


def propagate_deletion(nodes: list[dict[str, Any]], edges: list[tuple[str, str]], *, roots: set[str]) -> dict[str, Any]:
    children: dict[str, list[str]] = defaultdict(list)
    for parent, child in edges:
        children[parent].append(child)
    reached = set(roots)
    queue = list(roots)
    while queue:
        parent = queue.pop(0)
        for child in children.get(parent, []):
            if child not in reached:
                reached.add(child)
                queue.append(child)
    states = {}
    for node in nodes:
        node_id = node["id"]
        kind = node["kind"]
        if node_id not in reached:
            states[node_id] = "unverified_descendant_retained"
        elif kind in {"source_row"}:
            states[node_id] = "deleted"
        elif kind in {"cache", "vcm_index", "retrieval_index"}:
            states[node_id] = "purged"
        elif kind in {"report", "publication"}:
            states[node_id] = "retraction_required"
        else:
            states[node_id] = "revoked_or_quarantined"
    unverified = sorted(node_id for node_id, state in states.items() if state == "unverified_descendant_retained")
    return {
        "closed": len(reached) == len(nodes),
        "reached_count": len(reached),
        "node_count": len(nodes),
        "states": states,
        "unverified_descendants": unverified,
    }


def recursive_synthetic_diagnostics(*, metrics: Counter[str], class_counts: Counter[str], depth_counts: Counter[str], family_counts: Counter[str]) -> dict[str, Any]:
    total = sum(class_counts.values())
    synthetic = sum(count for name, count in class_counts.items() if name in {"private_synthetic", "teacher_distillation", "materialized_synthetic"})
    unknown = int(depth_counts.get("unknown") or 0)
    sorted_families = sorted(family_counts.items(), key=lambda item: (item[1], item[0]))
    tail_count = sum(1 for _family, count in sorted_families if count <= 2)
    return {
        "candidate_count": total,
        "synthetic_candidate_count": synthetic,
        "synthetic_share": round(synthetic / max(1, total), 6),
        "provenance_class_counts": dict(sorted(class_counts.items())),
        "synthetic_depth_counts": dict(sorted(depth_counts.items())),
        "unknown_lineage_depth_count": unknown,
        "parent_lineage_edge_count": int(metrics.get("parent_lineage_edges") or 0),
        "family_count": len(family_counts),
        "tail_family_count": tail_count,
        "largest_family_share": round(max(family_counts.values(), default=0) / max(1, total), 6),
        "collapse_risk_flags": [
            flag for flag, active in {
                "synthetic_share_above_0_8": synthetic / max(1, total) > 0.8,
                "unknown_lineage_depth_present": unknown > 0,
                "single_family_above_0_5": max(family_counts.values(), default=0) / max(1, total) > 0.5,
            }.items() if active
        ],
        "non_claim": "Distribution diagnostics are not proof of model collapse or data quality.",
    }


def build_viea_records(
    summary: dict[str, Any],
    ledger: dict[str, Any],
    continual: dict[str, Any],
    deletion: dict[str, Any],
    adversary: dict[str, Any],
    full_state: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "source_surface": "teacher_and_data_governance",
        "evidence_ref": "reports/training_data_lineage_audit.json",
        "support_state": "SUPPORTED" if summary["hard_gap_count"] == 0 else "BLOCKED",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    record_specs = [
        ("data_admission_receipt", {"receipt_count": summary["candidate_receipt_count"], "admitted_count": summary["admitted_candidate_count"], "ledger_sha256": ledger["sha256"]}),
        ("data_lineage_edge", {"edge_count": summary["lineage_edge_count"], "lineage_complete": True}),
        ("license_receipt", {"candidate_receipt_count": summary["candidate_receipt_count"], "policy": "per_candidate_license_or_source_authority"}),
        ("leakage_audit_receipt", {"exact_overlap_count": summary["exact_public_overlap_candidate_count"], "semantic_overlap_count": summary["semantic_public_overlap_candidate_count"], "adversary_passed": adversary["passed_count"] == adversary["case_count"]}),
        ("verifier_acceptance", {"teacher_candidate_count": summary["teacher_candidate_count"], "runtime_direct_serving_allowed": False}),
        ("data_lifecycle_policy_decision", {"recommended_policy": continual.get("recommended_policy"), "workload_hash": continual.get("workload_hash")}),
        ("continual_learning_comparison", {"policy_count": continual.get("policy_count"), "comparison_ready": continual.get("comparison_ready")}),
        ("descendant_deletion_closure_receipt", {"artifact_kind_count": deletion["artifact_kind_count"], "positive_closed": deletion["positive_fixture_closed"], "invalid_rejected": deletion["expected_invalid_fixture_rejected"]}),
        ("full_state_update_transaction", {
            "state": full_state.get("trigger_state"),
            "artifact_kind_count": (full_state.get("summary") or {}).get("artifact_kind_count"),
            "best_checkpoint_id": (full_state.get("summary") or {}).get("best_checkpoint_id"),
            "final_checkpoint_id": (full_state.get("summary") or {}).get("final_checkpoint_id"),
            "exact_rollback": bool((full_state.get("rollback") or {}).get("exact_pre_state_restored")),
            "behavioral_unlearning_claim_allowed": False,
        }),
        ("failure_boundary", {"terminal": summary["hard_gap_count"] > 0, "structured_non_solved": summary["hard_gap_count"] > 0}),
    ]
    return [
        {**common, "record_type": record_type, "record_id": stable_id(record_type, payload), **payload}
        for record_type, payload in record_specs
    ]


def reusable_bundle(
    *,
    input_digest: str,
    ledger_path: Path,
    started: float,
    migrated_teacher_receipts: int = 0,
) -> dict[str, Any]:
    prior = read_json(DEFAULT_OUT)
    if prior.get("policy") != POLICY or prior.get("input_digest") != input_digest:
        return {}
    if prior.get("trigger_state") not in {"GREEN", "YELLOW"} or as_list(prior.get("hard_gaps")):
        return {}
    ledger = as_dict(prior.get("candidate_receipt_ledger"))
    if str(ledger.get("path") or "") != rel(ledger_path):
        return {}
    if not ledger_path.exists() or file_sha256(ledger_path) != str(ledger.get("sha256") or ""):
        return {}
    reused = json.loads(json.dumps(prior))
    reused["created_utc"] = now()
    reused["summary"]["runtime_ms"] = int((time.perf_counter() - started) * 1000)
    reused["summary"].pop("cache_reused", None)
    reused["summary"]["report_cache_reused"] = True
    reused["summary"]["ledger_schema_migrated_teacher_receipt_count"] = migrated_teacher_receipts
    reused["candidate_receipt_ledger"]["cache_reused"] = True
    return reused


def recover_ledger_state(
    path: Path,
    *,
    source_rows: list[dict[str, Any]],
    teacher_manifest_path: Path,
    contamination_digest: str,
    expected_count: int,
) -> dict[str, Any]:
    """Replay a complete content-bound ledger instead of rescanning source text.

    This is intentionally stricter than the report cache. Every receipt remains
    bound to the current source digest and public contamination index, and
    duplicate or malformed identities invalidate the whole recovery.
    """
    if not path.exists() or expected_count <= 0:
        return {}
    source_digests = {
        str(row.get("source_id") or ""): str(row.get("sha256") or "")
        for row in source_rows
        if row.get("source_id") and row.get("sha256")
    }
    teacher_sha = file_sha256(teacher_manifest_path)
    teacher_source_id = stable_id("teacher-manifest", teacher_sha)
    if teacher_sha:
        source_digests[teacher_source_id] = teacher_sha

    metrics: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    depths: Counter[str] = Counter()
    families: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    workload: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    receipt_ids: set[str] = set()
    processed = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                receipt = json.loads(line)
                candidate_id = str(receipt.get("candidate_id") or "")
                receipt_id = str(receipt.get("receipt_id") or "")
                source_id = str(receipt.get("source_id") or "")
                contamination = as_dict(receipt.get("contamination"))
                if (
                    receipt.get("record_type") != "data_admission_receipt"
                    or receipt.get("policy") != POLICY
                    or receipt.get("raw_payload_stored") is not False
                    or receipt.get("decision") not in {"admit", "quarantine", "reject"}
                    or not candidate_id
                    or not receipt_id
                    or candidate_id in candidate_ids
                    or receipt_id in receipt_ids
                    or source_id not in source_digests
                    or str(receipt.get("source_sha256") or "") != source_digests[source_id]
                    or contamination.get("public_index_digest") != contamination_digest
                    or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("row_sha256") or ""))
                ):
                    return {}
                candidate_ids.add(candidate_id)
                receipt_ids.add(receipt_id)
                processed += 1
                update_metrics(metrics, classes, depths, families, source_counts, receipt)
                workload.append(workload_item(receipt, processed))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    if processed != expected_count or sum(source_counts.values()) != expected_count:
        return {}
    return {
        "processed": processed,
        "metrics": metrics,
        "class_counts": classes,
        "depth_counts": depths,
        "family_counts": families,
        "source_receipt_counts": source_counts,
        "workload": workload,
        "ledger_receipt": {
            "path": rel(path),
            "exists": True,
            "receipt_count": processed,
            "decision_counts": {
                key: int(metrics.get(key) or 0)
                for key in ("admit", "quarantine", "reject")
                if metrics.get(key)
            },
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "compression": "gzip_jsonl",
            "payload_text_stored": False,
            "replay_valid": True,
        },
    }


def incremental_rebuild_ledger(
    path: Path,
    *,
    source_rows: list[dict[str, Any]],
    teacher_manifest: dict[str, Any],
    teacher_manifest_path: Path,
    contamination_index: dict[str, Any],
    expected_count: int,
) -> dict[str, Any]:
    """Reuse valid source groups and rescan only changed candidate sources."""
    if not path.exists() or expected_count <= 0:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_candidates: set[str] = set()
    seen_receipts: set[str] = set()
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                receipt = json.loads(line)
                candidate_id = str(receipt.get("candidate_id") or "")
                receipt_id = str(receipt.get("receipt_id") or "")
                contamination = as_dict(receipt.get("contamination"))
                if (
                    receipt.get("record_type") != "data_admission_receipt"
                    or receipt.get("policy") != POLICY
                    or receipt.get("raw_payload_stored") is not False
                    or receipt.get("decision") not in {"admit", "quarantine", "reject"}
                    or not candidate_id
                    or not receipt_id
                    or candidate_id in seen_candidates
                    or receipt_id in seen_receipts
                    or contamination.get("public_index_digest") != contamination_index["digest"]
                    or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("row_sha256") or ""))
                ):
                    return {}
                seen_candidates.add(candidate_id)
                seen_receipts.add(receipt_id)
                grouped[str(receipt.get("source_id") or "")].append(receipt)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}

    metrics: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    depths: Counter[str] = Counter()
    families: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    workload: list[dict[str, Any]] = []
    reused_source_count = 0
    rescanned_source_count = 0
    reused_receipt_count = 0
    rescanned_receipt_count = 0
    processed = 0
    temporary = tempfile.NamedTemporaryFile(
        prefix="data-admission-incremental-",
        suffix=".jsonl.gz",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=6) as handle:
            for source in source_rows:
                source_id = str(source.get("source_id") or "")
                prior = grouped.get(source_id, [])
                expected_source_count = int(source.get("row_count") or 0)
                reusable = bool(
                    prior
                    and len(prior) == expected_source_count
                    and all(str(row.get("source_sha256") or "") == str(source.get("sha256") or "") for row in prior)
                )
                if reusable:
                    receipts = prior
                    reused_source_count += 1
                    reused_receipt_count += len(receipts)
                else:
                    receipts = [
                        candidate_receipt(
                            row,
                            source=source,
                            row_index=row_index,
                            contamination_index=contamination_index,
                        )
                        for row_index, row in enumerate(iter_rows(resolve(str(source.get("path") or ""))))
                    ]
                    if len(receipts) != expected_source_count:
                        return {}
                    rescanned_source_count += 1
                    rescanned_receipt_count += len(receipts)
                for receipt in receipts:
                    handle.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
                    processed += 1
                    update_metrics(metrics, classes, depths, families, source_counts, receipt)
                    workload.append(workload_item(receipt, processed))

            teacher_sha = file_sha256(teacher_manifest_path)
            teacher_source_id = stable_id("teacher-manifest", teacher_sha)
            teacher_rows = [row for row in as_list(teacher_manifest.get("rows")) if isinstance(row, dict)]
            prior_teacher = grouped.get(teacher_source_id, [])
            teacher_reusable = bool(
                prior_teacher
                and len(prior_teacher) == len(teacher_rows)
                and all(str(row.get("source_sha256") or "") == teacher_sha for row in prior_teacher)
            )
            if teacher_reusable:
                teacher_receipts = prior_teacher
                reused_source_count += 1
                reused_receipt_count += len(teacher_receipts)
            else:
                teacher_receipts = [
                    teacher_candidate_receipt(
                        row,
                        row_index=row_index,
                        manifest=teacher_manifest,
                        manifest_path=teacher_manifest_path,
                        contamination_index=contamination_index,
                    )
                    for row_index, row in enumerate(teacher_rows)
                ]
                rescanned_source_count += 1
                rescanned_receipt_count += len(teacher_receipts)
            for receipt in teacher_receipts:
                handle.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
                processed += 1
                update_metrics(metrics, classes, depths, families, source_counts, receipt)
                workload.append(workload_item(receipt, processed))
        if processed != expected_count:
            return {}
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "processed": processed,
        "metrics": metrics,
        "class_counts": classes,
        "depth_counts": depths,
        "family_counts": families,
        "source_receipt_counts": source_counts,
        "workload": workload,
        "reused_source_count": reused_source_count,
        "rescanned_source_count": rescanned_source_count,
        "reused_receipt_count": reused_receipt_count,
        "rescanned_receipt_count": rescanned_receipt_count,
        "ledger_receipt": {
            "path": rel(path),
            "exists": True,
            "receipt_count": processed,
            "decision_counts": {
                key: int(metrics.get(key) or 0)
                for key in ("admit", "quarantine", "reject")
                if metrics.get(key)
            },
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "compression": "gzip_jsonl",
            "payload_text_stored": False,
            "replay_valid": True,
        },
    }


def upgrade_teacher_lineage_receipts(
    path: Path,
    *,
    teacher_manifest: dict[str, Any],
    teacher_manifest_path: Path,
    contamination_index: dict[str, Any],
) -> int:
    """Repair the pre-lineage teacher receipt shape under strict identity checks."""
    teacher_rows = [row for row in as_list(teacher_manifest.get("rows")) if isinstance(row, dict)]
    if not path.exists() or not teacher_rows:
        return 0
    replacements = {
        index: teacher_candidate_receipt(
            row,
            row_index=index,
            manifest=teacher_manifest,
            manifest_path=teacher_manifest_path,
            contamination_index=contamination_index,
        )
        for index, row in enumerate(teacher_rows)
    }
    teacher_source_id = stable_id("teacher-manifest", file_sha256(teacher_manifest_path))
    migration_needed = False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                receipt = json.loads(line)
                if receipt.get("source_id") != teacher_source_id:
                    continue
                row_index = receipt.get("row_index")
                replacement = replacements.get(row_index) if isinstance(row_index, int) else None
                if replacement is None or any(
                    receipt.get(key) != replacement.get(key)
                    for key in ("candidate_id", "row_sha256", "source_sha256", "decision")
                ):
                    return 0
                provenance = as_dict(receipt.get("provenance"))
                migration_needed = migration_needed or (
                    provenance.get("lineage_complete") is not True
                    or not provenance.get("parent_ref_hashes")
                )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0
    if not migration_needed:
        return 0

    temporary = tempfile.NamedTemporaryFile(
        prefix="data-admission-lineage-migration-",
        suffix=".jsonl.gz",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    migrated = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source_handle, gzip.open(
            temporary_path, "wt", encoding="utf-8", compresslevel=6
        ) as target_handle:
            for line in source_handle:
                receipt = json.loads(line)
                if receipt.get("source_id") == teacher_source_id:
                    row_index = receipt.get("row_index")
                    replacement = replacements.get(row_index) if isinstance(row_index, int) else None
                    if replacement is None:
                        return 0
                    receipt = replacement
                    migrated += 1
                target_handle.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
        if migrated:
            os.replace(temporary_path, path)
        return migrated
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def audit_ledger(path: Path) -> dict[str, Any]:
    count = 0
    decisions: Counter[str] = Counter()
    valid = True
    if path.exists():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    count += 1
                    decisions[str(row.get("decision") or "unknown")] += 1
                    if row.get("record_type") != "data_admission_receipt" or row.get("raw_payload_stored") is not False:
                        valid = False
        except (OSError, json.JSONDecodeError):
            valid = False
    return {
        "path": rel(path),
        "exists": path.exists(),
        "receipt_count": count,
        "decision_counts": dict(sorted(decisions.items())),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size if path.exists() else 0,
        "compression": "gzip_jsonl",
        "payload_text_stored": False,
        "replay_valid": bool(path.exists() and count > 0 and valid),
    }


def load_admitted_candidate_hashes(admission: dict[str, Any]) -> set[str]:
    lineage = as_dict(admission.get("candidate_lineage"))
    ledger = as_dict(lineage.get("candidate_receipt_ledger"))
    path = resolve(str(ledger.get("path") or ""))
    if not path.exists() or file_sha256(path) != str(ledger.get("sha256") or ""):
        return set()
    hashes = set()
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("decision") == "admit" and row.get("row_sha256"):
                    hashes.add(str(row["row_sha256"]))
    except (OSError, json.JSONDecodeError):
        return set()
    return hashes


def update_metrics(metrics: Counter[str], classes: Counter[str], depths: Counter[str], families: Counter[str], source_counts: Counter[str], receipt: dict[str, Any]) -> None:
    decision = str(receipt.get("decision") or "unknown")
    metrics[decision] += 1
    contamination = as_dict(receipt.get("contamination"))
    if int(contamination.get("exact_overlap_count") or 0):
        metrics["exact_overlap"] += 1
        if decision == "admit":
            metrics["admitted_exact_public_overlap"] += 1
    if int(contamination.get("semantic_overlap_count") or 0):
        metrics["semantic_overlap"] += 1
        if decision == "admit":
            metrics["admitted_semantic_public_overlap"] += 1
    metrics["public_flag"] += int(receipt.get("public_flag_detected") is True)
    metrics["raw_user"] += int(receipt.get("raw_user_text_detected") is True)
    metrics["fallback_marker"] += int(receipt.get("fallback_marker_detected") is True)
    metrics["lineage_edges"] += len(as_list(receipt.get("lineage_edges")))
    for reason in as_list(receipt.get("decision_reasons")):
        metrics[f"reason:{reason}"] += 1
    provenance = as_dict(receipt.get("provenance"))
    metrics["parent_lineage_edges"] += len(as_list(provenance.get("parent_ref_hashes")))
    classes[str(provenance.get("class") or "unknown")] += 1
    depth = provenance.get("synthetic_depth")
    depths[str(depth) if isinstance(depth, int) else "unknown"] += 1
    families[str(receipt.get("family") or "unknown")] += 1
    source_counts[str(receipt.get("source_id") or "unknown")] += 1


def workload_item(receipt: dict[str, Any], order: int) -> dict[str, Any]:
    provenance = as_dict(receipt.get("provenance"))
    return {
        "candidate_id": receipt.get("candidate_id"),
        "family": receipt.get("family") or "unknown",
        "decision": receipt.get("decision"),
        "order": order,
        "lineage_complete": provenance.get("lineage_complete") is True,
        "synthetic_depth": int(provenance.get("synthetic_depth") or 0),
        "simulated_revoked": stable_mod(str(receipt.get("candidate_id") or ""), 101) == 0,
    }


def provenance_profile(row: dict[str, Any], source: dict[str, Any]) -> tuple[str, int | None, list[str]]:
    source_kind = str(source.get("source_kind") or "")
    flat = flatten(row)
    provenance = as_dict(row.get("provenance"))
    parent_values = []
    for key, value in {**flat, **provenance}.items():
        lowered = str(key).lower()
        if any(token in lowered for token in ("materialized_from", "source_jsonl", "parent", "teacher_calls")) and value:
            parent_values.append(stable_id("lineage-parent", str(value)))
    parents = sorted(set(parent_values))
    serialized = json.dumps(row, sort_keys=True).lower()
    if source_kind == "teacher_distillation":
        return "teacher_distillation", 1, parents
    if source_kind == "dogfood_metadata":
        return "private_dogfood_metadata", 0, parents
    if source_kind == "open_public_training_rows":
        return "licensed_open_data", 0, parents
    if "materialized_from" in serialized:
        return "materialized_synthetic", 2, parents
    if any(token in serialized for token in ("private_residual_generated", "synthetic", "generated_train_only")):
        return "private_synthetic", 1, parents
    if source_kind in {"private_training_rows", "private_code_curriculum"}:
        if (
            isinstance(row.get("provenance"), dict)
            or any(flat.get(key) for key in ("policy", "source_id", "benchmark_evidence_level"))
            or str(source.get("provenance_status") or "") in {"provenance_tags_present", "project_internal_path_provenance"}
        ):
            return "private_project_data", 0, parents
        return "unknown", None, parents
    return "unknown", None, parents


def evaluation_ref_hashes(row: dict[str, Any]) -> list[str]:
    flat = flatten(row)
    values = []
    for key, value in flat.items():
        if (
            any(token in key.lower() for token in ("test", "verifier", "evaluation", "residual"))
            and value is not None
            and value != ""
            and value is not False
        ):
            values.append(stable_id("evaluation-ref", key, value))
    return sorted(set(values))[:16]


def family_label(row: dict[str, Any]) -> str:
    flat = flatten(row)
    for key in ("broad_private_family_v1", "targeted_private_residual_family_v3", "category", "assistant_lane", "task_family"):
        value = flat.get(key)
        if value:
            return str(value)[:160]
    return "unknown"


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
        return
    payload = read_json(path)
    for key in ("rows", "items", "sources", "capsules", "repos"):
        values = payload.get(key)
        if isinstance(values, list):
            for row in values:
                if isinstance(row, dict):
                    yield row
            return
    if payload:
        yield payload


def extract_texts(value: Any, key: str = "") -> list[str]:
    out = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            out.extend(extract_texts(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            out.extend(extract_texts(child, key))
    elif isinstance(value, str) and key.split(".")[-1].lower() in TEXT_KEYS and len(value.strip()) >= 8:
        out.append(value)
    return out


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(child, child_key))
            out[str(key)] = child
    elif prefix:
        out[prefix] = value
    return out


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def text_sha256(value: Any) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def row_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def token_hashes(text: str) -> frozenset[int]:
    tokens = re.findall(r"[a-z_][a-z0-9_]*|\d+|[^\w\s]", normalize_text(text))
    return frozenset(int(hashlib.sha256(token.encode()).hexdigest()[:16], 16) for token in tokens)


def minhash(tokens: frozenset[int]) -> tuple[int, ...]:
    if not tokens:
        return tuple(0 for _ in MINHASH_SEEDS)
    return tuple(min((((token ^ seed) * 0x9E3779B185EBCA87) & MASK64) for token in tokens) for seed in MINHASH_SEEDS)


def minhash_bands(signature: tuple[int, ...]) -> list[tuple[int, int, int]]:
    return [(index, value, 0) for index, value in enumerate(signature)]


def stable_mod(value: str, modulus: int) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % max(1, modulus)


def stable_id(prefix: str, *parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:20]}"


def file_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_license(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def first_value(flat: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = flat.get(key)
        if value is not None and value != "":
            return value
    return None


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "enabled", "allowed"}


def compact_console_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "summary": report.get("summary"),
        "candidate_receipt_ledger": report.get("candidate_receipt_ledger"),
        "hard_gaps": report.get("hard_gaps"),
        "warnings": report.get("warnings"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = as_dict(report.get("summary"))
    continual = as_dict(report.get("continual_learning_policy_comparison"))
    lines = [
        "# Training Data Lineage Audit",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- candidate receipts: `{summary.get('candidate_receipt_count', 0)}`",
        f"- admitted / quarantined / rejected: `{summary.get('admitted_candidate_count', 0)}` / `{summary.get('quarantined_candidate_count', 0)}` / `{summary.get('rejected_candidate_count', 0)}`",
        f"- exact / semantic public overlap candidates: `{summary.get('exact_public_overlap_candidate_count', 0)}` / `{summary.get('semantic_public_overlap_candidate_count', 0)}`",
        f"- adversary controls: `{summary.get('adversary_passed_count', 0)}/{summary.get('adversary_case_count', 0)}`",
        f"- continual policies: `{continual.get('policy_count', 0)}`; recommended `{continual.get('recommended_policy', 'none')}`",
        f"- deletion artifact kinds: `{summary.get('deletion_artifact_kind_count', 0)}`",
        f"- full-state causality: `{summary.get('full_state_causality_state', 'missing')}`; artifact kinds `{summary.get('full_state_artifact_kind_count', 0)}`; exact rollback `{summary.get('full_state_exact_rollback', False)}`",
        f"- ledger: `{as_dict(report.get('candidate_receipt_ledger')).get('path', '')}`",
        "",
        "This is data-governance evidence, not model training or capability evidence.",
    ]
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    value = resolve(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except ValueError:
        return str(value)


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
