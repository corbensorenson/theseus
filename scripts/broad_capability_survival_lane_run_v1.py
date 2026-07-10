#!/usr/bin/env python3
"""Materialize and optionally run the broad capability survival-lane comparator.

This is the first training-facing consumer of training_data_admission_v1 and
broad_capability_curriculum_v1. It creates a broad private train/eval pair and
runs the existing matched neural-seed comparator with the transformer/hybrid
path treated as the survival lane and SymLiquid as a comparator only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_code_proposer_comparator import run_comparator  # noqa: E402
import training_data_lineage_audit  # noqa: E402


DEFAULT_ADMISSION = ROOT / "reports" / "training_data_admission_v1.json"
DEFAULT_CURRICULUM = ROOT / "reports" / "broad_capability_curriculum_v1.json"
DEFAULT_BASE_CONFIG = ROOT / "configs" / "neural_seed_code_proposer_comparator.json"
DEFAULT_TRAIN = ROOT / "reports" / "broad_capability_survival_lane_v1_train.jsonl"
DEFAULT_EVAL = ROOT / "reports" / "broad_capability_survival_lane_v1_eval.jsonl"
DEFAULT_CONFIG = ROOT / "reports" / "broad_capability_survival_lane_v1_comparator_config.json"
DEFAULT_CANDIDATES = ROOT / "reports" / "broad_capability_survival_lane_v1_candidates.jsonl"
DEFAULT_COMPARATOR = ROOT / "reports" / "broad_capability_survival_lane_v1_comparator.json"
DEFAULT_COMPARATOR_MD = ROOT / "reports" / "broad_capability_survival_lane_v1_comparator.md"
DEFAULT_OUT = ROOT / "reports" / "broad_capability_survival_lane_run_v1.json"
DEFAULT_MD = ROOT / "reports" / "broad_capability_survival_lane_run_v1.md"
DEFAULT_STS_POLICY = ROOT / "configs" / "sts_broad_survival_policy_v1.json"
DEFAULT_VCM_CONTEXTS = ROOT / "reports" / "vcm_task_contexts.json"
DEFAULT_VCM_FEATURE_POLICY = ROOT / "configs" / "vcm_broad_survival_feature_policy_v1.json"

PUBLIC_FLAG_KEYS = [
    "public_benchmark",
    "public_benchmark_row",
    "public_benchmark_solutions_included",
    "public_prompts_included",
    "public_score_labels_included",
    "public_tests_included",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", default=rel(DEFAULT_ADMISSION))
    parser.add_argument("--curriculum", default=rel(DEFAULT_CURRICULUM))
    parser.add_argument("--base-config", default=rel(DEFAULT_BASE_CONFIG))
    parser.add_argument("--train-out", default=rel(DEFAULT_TRAIN))
    parser.add_argument("--eval-out", default=rel(DEFAULT_EVAL))
    parser.add_argument("--config-out", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--candidate-manifest-out", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--comparator-out", default=rel(DEFAULT_COMPARATOR))
    parser.add_argument("--comparator-markdown-out", default=rel(DEFAULT_COMPARATOR_MD))
    parser.add_argument("--sts-policy", default=rel(DEFAULT_STS_POLICY))
    parser.add_argument("--vcm-contexts", default=rel(DEFAULT_VCM_CONTEXTS))
    parser.add_argument("--vcm-feature-policy", default=rel(DEFAULT_VCM_FEATURE_POLICY))
    parser.add_argument("--vcm-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--max-train-rows", type=int, default=4096)
    parser.add_argument("--max-eval-rows", type=int, default=192)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_and_maybe_run(args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps({
        "trigger_state": report.get("trigger_state"),
        "summary": report.get("summary"),
        "candidate_hash_filter": (report.get("materialization") or {}).get("candidate_hash_filter"),
        "failed_hard_gates": [
            row.get("name") for row in report.get("gates", [])
            if row.get("severity") == "hard" and row.get("passed") is not True
        ],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_and_maybe_run(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    admission = read_json(resolve(args.admission))
    curriculum = read_json(resolve(args.curriculum))
    base_config = read_json(resolve(args.base_config))
    sts_policy = read_json(resolve(args.sts_policy))
    vcm_contexts = read_json(resolve(args.vcm_contexts))
    vcm_feature_policy = read_json(resolve(args.vcm_feature_policy))
    materialized = materialize_rows(
        admission=admission,
        curriculum=curriculum,
        max_train_rows=max(1, int(args.max_train_rows)),
        max_eval_rows=max(1, int(args.max_eval_rows)),
        seed=int(args.seed),
        vcm_contexts=vcm_contexts,
        vcm_feature_policy=vcm_feature_policy,
        vcm_mode=str(args.vcm_mode),
    )
    train_path = resolve(args.train_out)
    eval_path = resolve(args.eval_out)
    write_jsonl(train_path, materialized["train_rows"])
    write_jsonl(eval_path, materialized["eval_rows"])
    config = build_config(
        base_config,
        train_path=train_path,
        eval_path=eval_path,
        seed=int(args.seed),
        epochs=max(1, int(args.epochs)),
        max_train_rows=len(materialized["train_rows"]),
        max_eval_rows=len(materialized["eval_rows"]),
        sts_policy=sts_policy,
        vcm_summary=materialized["vcm_summary"],
    )
    config_path = resolve(args.config_out)
    write_json(config_path, config)

    comparator_report: dict[str, Any] = {}
    if args.execute:
        comparator_started = time.perf_counter()
        comparator_report = run_comparator(
            config,
            rel(config_path),
            rel(resolve(args.candidate_manifest_out)),
            comparator_started,
        )
        write_json(resolve(args.comparator_out), comparator_report)
        write_text(resolve(args.comparator_markdown_out), render_comparator_markdown(comparator_report))

    gates = build_gates(args, admission, curriculum, materialized, comparator_report)
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    trigger_state = "GREEN" if not hard_failed else "RED"
    if trigger_state == "GREEN" and (not args.execute or comparator_report.get("trigger_state") == "YELLOW"):
        trigger_state = "YELLOW"
    return {
        "policy": "project_theseus_broad_capability_survival_lane_run_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "execute": bool(args.execute),
        "summary": {
            "train_rows": len(materialized["train_rows"]),
            "eval_rows": len(materialized["eval_rows"]),
            "train_source_count": len(materialized["train_sources"]),
            "eval_source_count": len(materialized["eval_sources"]),
            "train_sha256": sha256_file(train_path),
            "eval_sha256": sha256_file(eval_path),
            "candidate_manifest": rel(resolve(args.candidate_manifest_out)),
            "comparator_state": comparator_report.get("trigger_state") if comparator_report else "PLANNED",
            "transformer_sts_on_pass_rate": get_path(
                comparator_report,
                ["comparisons", "by_arm", "transformer_control", "sts_on_verifier_pass_rate"],
            ),
            "symliquid_sts_on_pass_rate": get_path(
                comparator_report,
                ["comparisons", "by_arm", "symliquid_style", "sts_on_verifier_pass_rate"],
            ),
            "symliquid_minus_transformer": get_path(
                comparator_report,
                ["comparisons", "symliquid_minus_transformer_sts_on_verifier_pass_rate"],
            ),
            "winner_by_sts_on": get_path(comparator_report, ["comparisons", "winner_by_sts_on_verifier_pass_rate"]),
            "public_benchmark_training_rows": 0,
            "fallback_return_count": 0,
            "external_inference_calls": 0,
            "teacher_used": False,
            "model_promotion_allowed": False,
            "sts_policy_action": sts_policy.get("action") if sts_policy else "no_policy_found",
            "sts_policy_applied": bool(config.get("sts_policy_applied")),
            "effective_sts_on_fields": get_path(config, ["text_views", "sts_on"], []),
            "vcm_mode": materialized["vcm_summary"].get("mode"),
            "vcm_context_active": materialized["vcm_summary"].get("active"),
            "vcm_rows_with_context": materialized["vcm_summary"].get("rows_with_context"),
            "vcm_unique_context_hashes": materialized["vcm_summary"].get("unique_context_hashes"),
        },
        "inputs": {
            "admission": rel(resolve(args.admission)),
            "curriculum": rel(resolve(args.curriculum)),
            "base_config": rel(resolve(args.base_config)),
            "sts_policy": rel(resolve(args.sts_policy)),
            "vcm_contexts": rel(resolve(args.vcm_contexts)),
            "vcm_feature_policy": rel(resolve(args.vcm_feature_policy)),
        },
        "artifacts": {
            "train_jsonl": rel(train_path),
            "eval_jsonl": rel(eval_path),
            "comparator_config": rel(config_path),
            "candidate_manifest": rel(resolve(args.candidate_manifest_out)),
            "comparator_report": rel(resolve(args.comparator_out)),
            "markdown": rel(resolve(args.markdown_out)),
        },
        "sts_policy": {
            "loaded": bool(sts_policy),
            "source": rel(resolve(args.sts_policy)),
            "applied": bool(config.get("sts_policy_applied")),
            "action": sts_policy.get("action") if sts_policy else "no_policy_found",
            "evidence": sts_policy.get("evidence") if isinstance(sts_policy.get("evidence"), dict) else {},
        },
        "vcm_context": materialized["vcm_summary"],
        "materialization": {
            "train_sources": materialized["train_sources"],
            "eval_sources": materialized["eval_sources"],
            "train_family_counts": materialized["train_family_counts"],
            "eval_family_counts": materialized["eval_family_counts"],
            "candidate_hash_filter": materialized["candidate_hash_filter"],
        },
        "architecture_policy": {
            "survival_lane": "transformer_hybrid_structural_student",
            "symliquid_role": "bounded_matched_discovery_comparator_only",
            "public_calibration_allowed": False,
            "public_benchmark_training_allowed": False,
        },
        "comparator_report": compact_comparator(comparator_report),
        "gates": gates,
        "score_semantics": (
            "Private broad survival-lane materialization and optional matched comparator only. It does not "
            "train on public benchmark payloads, call a teacher, run public calibration, or promote a model."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def materialize_rows(
    *,
    admission: dict[str, Any],
    curriculum: dict[str, Any],
    max_train_rows: int,
    max_eval_rows: int,
    seed: int,
    vcm_contexts: dict[str, Any] | None = None,
    vcm_feature_policy: dict[str, Any] | None = None,
    vcm_mode: str = "auto",
) -> dict[str, Any]:
    admitted_candidate_hashes = training_data_lineage_audit.load_admitted_candidate_hashes(admission)
    units = [row for row in curriculum.get("curriculum_units", []) if isinstance(row, dict)]
    train_rows = []
    train_sources = []
    per_unit = max(1, math.ceil(max_train_rows / max(1, len(units))))
    for unit in units:
        path = resolve(str(unit.get("path") or ""))
        rows = clean_training_rows(read_jsonl(path), admitted_candidate_hashes=admitted_candidate_hashes)
        rows = deterministic_sample(rows, min(per_unit, int(unit.get("row_budget") or per_unit)), seed + stable_int(str(unit.get("unit_id"))))
        if rows:
            train_sources.append({"path": rel(path), "rows": len(rows), "unit_id": unit.get("unit_id")})
            train_rows.extend(rows)
    train_rows = deterministic_sample(dedupe_rows(train_rows), max_train_rows, seed + 17)

    eval_sources = []
    eval_rows = []
    heldout_sources = [
        row
        for row in admission.get("source_admissions", [])
        if isinstance(row, dict)
        and row.get("training_use") == "heldout_eval_only"
        and not row.get("public_benchmark_payload_detected")
    ]
    per_eval = max(1, math.ceil(max_eval_rows / max(1, len(heldout_sources))))
    for source in heldout_sources:
        path = resolve(str(source.get("path") or ""))
        rows = clean_eval_rows(read_jsonl(path))
        rows = deterministic_sample(rows, per_eval, seed + stable_int(str(source.get("source_id"))))
        if rows:
            eval_sources.append({"path": rel(path), "rows": len(rows), "source_id": source.get("source_id")})
            eval_rows.extend(rows)
    eval_rows = deterministic_sample(dedupe_rows(eval_rows), max_eval_rows, seed + 101)
    vcm_summary = apply_vcm_contexts_to_materialized_rows(
        train_rows,
        eval_rows,
        vcm_contexts=vcm_contexts or {},
        vcm_feature_policy=vcm_feature_policy or {},
        vcm_mode=vcm_mode,
    )
    return {
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "train_sources": train_sources,
        "eval_sources": eval_sources,
        "train_family_counts": family_counts(train_rows),
        "eval_family_counts": family_counts(eval_rows),
        "vcm_summary": vcm_summary,
        "candidate_hash_filter": {
            "ready": bool(admitted_candidate_hashes),
            "admitted_hash_count": len(admitted_candidate_hashes),
            "selected_train_hashes_all_admitted": bool(train_rows) and all(
                training_data_lineage_audit.row_sha256(row) in admitted_candidate_hashes for row in train_rows
            ),
            "ledger": ((admission.get("candidate_lineage") or {}).get("candidate_receipt_ledger") or {}).get("path"),
        },
    }


def build_config(
    base_config: dict[str, Any],
    *,
    train_path: Path,
    eval_path: Path,
    seed: int,
    epochs: int,
    max_train_rows: int,
    max_eval_rows: int,
    sts_policy: dict[str, Any] | None = None,
    vcm_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["policy"] = "project_theseus_broad_capability_survival_lane_comparator_v1"
    config["purpose"] = (
        "Run a broader private transformer/hybrid survival-lane comparator from training_data_admission_v1 "
        "and broad_capability_curriculum_v1. SymLiquid remains a matched comparator only."
    )
    config["comparison_level"] = "broad_private_survival_lane_transformer_hybrid_vs_symliquid"
    config.setdefault("safety", {})
    config["safety"].update({
        "public_calibration_allowed": False,
        "public_training_allowed": False,
        "teacher_calls_allowed": False,
        "teacher_distillation_allowed": False,
        "network_fetch_allowed": False,
        "model_promotion_allowed": False,
        "long_unattended_training_allowed": False,
    })
    config["data"] = {
        "train_jsonl": rel(train_path),
        "eval_jsonl": rel(eval_path),
        "max_train_rows": max_train_rows,
        "max_eval_rows": max_eval_rows,
        "split_seed": seed,
        "require_private_flags": True,
        "forbidden_row_flags": PUBLIC_FLAG_KEYS,
        "training_target": "private_train_solution_body_template",
        "eval_tests_visible_to": "private_verifier_only",
        "eval_solution_visible_to_generator": False,
    }
    config.setdefault("matched_budget", {})
    config["matched_budget"].update({
        "budget_id": "broad_capability_survival_lane_v1_transformer_hybrid_private",
        "seeds": [seed],
        "max_sequence_tokens": 128,
        "max_vocab": 2048,
        "epochs": epochs,
        "batch_size": 64,
        "learning_rate": 0.0025,
        "weight_decay": 0.0001,
        "fanout_top_k": 4,
        "parameter_match_tolerance": 0.12,
        "trusted_parameter_match_tolerance": 0.08,
        "private_candidate_timeout_seconds": 4,
    })
    config.setdefault("arms", {})
    config["arms"].setdefault("transformer_control", {})
    config["arms"]["transformer_control"].update({
        "id": "broad_private_transformer_hybrid_structural_student",
        "kind": "torch_transformer_encoder_code_body_selector_survival_lane",
        "role": "primary survival-lane candidate-code proposer control",
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 160,
    })
    config["arms"].setdefault("symliquid_style", {})
    config["arms"]["symliquid_style"].update({
        "role": "matched discovery comparator only",
        "hidden_dim_candidates": [56, 64, 72, 80, 88, 96, 104, 112, 120, 128],
        "reservoir_dim_candidates": [56, 64, 72, 80, 88, 96, 104, 112, 120, 128],
        "hv_dim_candidates": [56, 64, 72, 80, 88, 96, 104, 112, 120, 128],
    })
    apply_sts_policy(config, sts_policy or {})
    apply_vcm_feature_config(config, vcm_summary or {})
    return config


def apply_sts_policy(config: dict[str, Any], sts_policy: dict[str, Any]) -> None:
    if not sts_policy:
        config["sts_policy_applied"] = False
        config["sts_policy_note"] = "no sts_broad_survival_policy_v1 file found"
        return
    action = str(sts_policy.get("action") or "")
    effective_fields = sts_policy.get("effective_sts_on_fields")
    if action == "disable_sts_for_broad_body_template_selector" and isinstance(effective_fields, list):
        config.setdefault("text_views", {})
        config["text_views"]["sts_on"] = [str(item) for item in effective_fields]
        config["sts_policy_applied"] = True
        config["sts_policy_note"] = (
            "STS-on view gated to evidence-backed effective fields for the broad body-template selector."
        )
        config["sts_policy_source"] = {
            "policy": sts_policy.get("policy"),
            "action": action,
            "disabled_or_deweighted_fields": sts_policy.get("disabled_or_deweighted_fields", []),
            "evidence": sts_policy.get("evidence", {}),
        }
        return
    config["sts_policy_applied"] = False
    config["sts_policy_note"] = f"policy action did not modify broad run: {action or 'missing_action'}"


def apply_vcm_feature_config(config: dict[str, Any], vcm_summary: dict[str, Any]) -> None:
    active = bool(vcm_summary.get("active"))
    config["vcm_context_feature"] = {
        "policy": "project_theseus_broad_survival_vcm_feature_v1",
        "active": active,
        "mode": vcm_summary.get("mode"),
        "source_snapshot": vcm_summary.get("snapshot"),
        "rows_with_context": vcm_summary.get("rows_with_context", 0),
        "task_family_ids": vcm_summary.get("task_family_ids", []),
        "feature_fields": [
            "vcm_context.task_family_id",
            "vcm_context.label",
            "vcm_context.selected_page_lanes",
            "vcm_context.selected_context_hash",
        ] if active else [],
        "score_semantics": "VCM context feature path only; no public training rows, teacher calls, or fallback returns.",
    }
    if not active:
        return
    text_views = config.setdefault("text_views", {})
    vcm_fields = config["vcm_context_feature"]["feature_fields"]
    for view in ["sts_off", "sts_on"]:
        fields = [str(item) for item in text_views.get(view, []) or []]
        for field in vcm_fields:
            if field not in fields:
                fields.append(field)
        text_views[view] = fields


def apply_vcm_contexts_to_materialized_rows(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    vcm_contexts: dict[str, Any],
    vcm_feature_policy: dict[str, Any],
    vcm_mode: str,
) -> dict[str, Any]:
    contexts = [row for row in vcm_contexts.get("task_contexts", []) if isinstance(row, dict)]
    ready_contexts = {str(row.get("task_family_id") or ""): row for row in contexts if row.get("ready")}
    policy_action = str(vcm_feature_policy.get("action") or "")
    disabled_by_policy = policy_action == "disable_vcm_for_broad_body_template_selector"
    active = vcm_mode == "on" or (
        vcm_mode == "auto" and bool(ready_contexts.get("code_training")) and not disabled_by_policy
    )
    summary = {
        "policy": "project_theseus_broad_survival_vcm_materialization_v1",
        "mode": "on" if active else "off",
        "requested_mode": vcm_mode,
        "feature_policy": vcm_feature_policy.get("policy") or "",
        "feature_policy_action": policy_action or "missing_policy",
        "disabled_by_policy": disabled_by_policy and vcm_mode != "on",
        "active": active,
        "contexts_loaded": len(contexts),
        "ready_contexts": len(ready_contexts),
        "snapshot": str(vcm_contexts.get("snapshot") or ""),
        "task_family_ids": sorted(ready_contexts),
        "rows_with_context": 0,
        "unique_context_hashes": [],
        "public_training_rows_written": int(vcm_contexts.get("public_training_rows_written") or 0),
        "external_inference_calls": int(vcm_contexts.get("external_inference_calls") or 0),
        "teacher_solving_calls": int(vcm_contexts.get("teacher_solving_calls") or 0),
        "fallback_return_count": int(vcm_contexts.get("fallback_return_count") or 0),
    }
    if not active:
        return summary
    hashes = set()
    rows_with_context = 0
    for row in train_rows + eval_rows:
        context = select_vcm_context_for_row(row, ready_contexts)
        if not context:
            continue
        feature = vcm_context_feature(row, context)
        row["vcm_context"] = feature
        hashes.add(str(feature.get("selected_context_hash") or ""))
        rows_with_context += 1
    summary["rows_with_context"] = rows_with_context
    summary["unique_context_hashes"] = sorted(hashes)
    return summary


def select_vcm_context_for_row(row: dict[str, Any], ready_contexts: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    family = str(row.get("broad_private_family_v1") or row.get("targeted_private_residual_family_v3") or row.get("category") or "").lower()
    category = str(row.get("category") or "").lower()
    text = f"{family} {category}"
    if "project_memory" in text or "context" in text:
        return ready_contexts.get("docs_project_state") or ready_contexts.get("code_training")
    if "runtime" in text or "mlx" in text or "metal" in text:
        return ready_contexts.get("runtime_mlx_metal") or ready_contexts.get("code_training")
    return ready_contexts.get("code_training")


def vcm_context_feature(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    pages = [page for page in context.get("selected_pages", []) if isinstance(page, dict)]
    selected = pages[:8]
    return {
        "policy": "project_theseus_vcm_context_feature_v1",
        "task_family_id": str(context.get("task_family_id") or ""),
        "label": str(context.get("label") or ""),
        "selected_context_hash": str(context.get("selected_context_hash") or ""),
        "selected_page_titles": [str(page.get("title") or "")[:120] for page in selected],
        "selected_page_lanes": sorted({str(page.get("lane") or "") for page in selected if str(page.get("lane") or "")}),
        "selected_page_sources": [str(page.get("source_path") or "")[:160] for page in selected],
        "row_family": str(row.get("broad_private_family_v1") or row.get("targeted_private_residual_family_v3") or ""),
        "row_category": str(row.get("category") or ""),
        "model_visible": True,
        "raw_user_text_included": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_output_count": 0,
    }


def build_gates(
    args: argparse.Namespace,
    admission: dict[str, Any],
    curriculum: dict[str, Any],
    materialized: dict[str, Any],
    comparator: dict[str, Any],
) -> list[dict[str, Any]]:
    train_rows = materialized["train_rows"]
    eval_rows = materialized["eval_rows"]
    return [
        gate("admission_not_red", admission.get("trigger_state") in {"GREEN", "YELLOW"}, admission.get("trigger_state"), "hard"),
        gate("curriculum_green", curriculum.get("trigger_state") == "GREEN", curriculum.get("trigger_state"), "hard"),
        gate("train_rows_materialized", len(train_rows) > 0, len(train_rows), "hard"),
        gate(
            "candidate_receipt_hash_filter_ready",
            materialized.get("candidate_hash_filter", {}).get("ready") is True,
            materialized.get("candidate_hash_filter"),
            "hard",
        ),
        gate(
            "all_training_rows_have_admitted_candidate_receipts",
            materialized.get("candidate_hash_filter", {}).get("selected_train_hashes_all_admitted") is True,
            materialized.get("candidate_hash_filter"),
            "hard",
        ),
        gate("eval_rows_materialized", len(eval_rows) > 0, len(eval_rows), "hard"),
        gate("train_rows_have_solution_body", all(bool(row.get("solution_body")) for row in train_rows), len(train_rows), "hard"),
        gate("eval_rows_have_tests", all(bool(row.get("tests")) for row in eval_rows), len(eval_rows), "hard"),
        gate("public_flags_false_train", public_flag_hits(train_rows) == 0, public_flag_hits(train_rows), "hard"),
        gate("public_flags_false_eval", public_flag_hits(eval_rows) == 0, public_flag_hits(eval_rows), "hard"),
        gate("fallback_return_zero", fallback_count(train_rows + eval_rows) == 0, fallback_count(train_rows + eval_rows), "hard"),
        gate("external_inference_zero", external_calls(train_rows + eval_rows) == 0, external_calls(train_rows + eval_rows), "hard"),
        gate(
            "comparator_ran_if_execute",
            (not args.execute) or comparator.get("trigger_state") in {"GREEN", "YELLOW"},
            comparator.get("trigger_state") if comparator else "PLANNED",
            "hard",
        ),
    ]


def clean_training_rows(rows: list[dict[str, Any]], *, admitted_candidate_hashes: set[str]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row_is_clean(row)
        and bool(row.get("solution_body"))
        and training_data_lineage_audit.row_sha256(row) in admitted_candidate_hashes
    ]


def clean_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row_is_clean(row) and bool(row.get("tests"))]


def row_is_clean(row: dict[str, Any]) -> bool:
    if any(bool(row.get(key)) for key in PUBLIC_FLAG_KEYS):
        return False
    if int(row.get("external_inference_calls") or 0) != 0:
        return False
    if contains_forbidden_fallback_return(row):
        return False
    return True


def deterministic_sample(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return list(rows)
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda item: stable_hash({"seed": seed, "idx": item[0], "id": row_id(item[1])}))
    return [row for _idx, row in indexed[:limit]]


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        key = row_id(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("source_task_id") or row.get("entry_point") or stable_hash(row)[:16])


def family_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        family = str(
            row.get("broad_private_family_v1")
            or row.get("targeted_private_residual_family_v3")
            or row.get("category")
            or "unknown"
        )
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def public_flag_hits(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows for key in PUBLIC_FLAG_KEYS if bool(row.get(key)))


def fallback_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if contains_forbidden_fallback_return(row))


def contains_forbidden_fallback_return(value: Any) -> bool:
    allowed_zero_telemetry = {
        "fallback_return_count",
        "fallback_returns_zero",
        "fallback_returns_allowed",
        "fallback_output_count",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in allowed_zero_telemetry and item in {0, 0.0, False, None}:
                continue
            if "fallback_return" in key_l or "fallback return" in key_l:
                return True
            if contains_forbidden_fallback_return(item):
                return True
        return False
    if isinstance(value, list):
        return any(contains_forbidden_fallback_return(item) for item in value)
    if isinstance(value, str):
        text = value.lower()
        return "fallback_return" in text or "fallback return" in text
    return False


def external_calls(rows: list[dict[str, Any]]) -> int:
    return sum(int(row.get("external_inference_calls") or 0) for row in rows)


def compact_comparator(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "trigger_state": report.get("trigger_state"),
        "summary": report.get("summary"),
        "comparisons": report.get("comparisons"),
        "failed_gates": [row for row in report.get("gates", []) if isinstance(row, dict) and not row.get("passed")],
    }


def render_comparator_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    comparisons = report.get("comparisons") if isinstance(report.get("comparisons"), dict) else {}
    lines = [
        "# Broad Capability Survival Lane Comparator",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- train_rows: `{summary.get('train_rows')}`",
        f"- eval_rows: `{summary.get('eval_rows')}`",
        f"- candidate_rows: `{summary.get('candidate_rows')}`",
        f"- winner_by_sts_on: `{comparisons.get('winner_by_sts_on_verifier_pass_rate')}`",
        f"- symliquid_minus_transformer: `{comparisons.get('symliquid_minus_transformer_sts_on_verifier_pass_rate')}`",
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


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Broad Capability Survival Lane Run v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- execute: `{report.get('execute')}`",
        f"- train_rows: `{summary.get('train_rows')}`",
        f"- eval_rows: `{summary.get('eval_rows')}`",
        f"- comparator_state: `{summary.get('comparator_state')}`",
        f"- winner_by_sts_on: `{summary.get('winner_by_sts_on')}`",
        f"- transformer_sts_on_pass_rate: `{summary.get('transformer_sts_on_pass_rate')}`",
        f"- symliquid_sts_on_pass_rate: `{summary.get('symliquid_sts_on_pass_rate')}`",
        f"- symliquid_minus_transformer: `{summary.get('symliquid_minus_transformer')}`",
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


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cursor = value
    for part in path:
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
