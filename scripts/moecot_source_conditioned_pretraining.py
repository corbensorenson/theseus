#!/usr/bin/env python3
"""Materialize licensed auxiliary objectives for canonical MoECOT arms.

This owner materializes code denoising and the KERC English objective views. It
does not train another model or grant capability credit to deterministic record
validation and compilation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from moecot_language_supervision import (
    BoundedRows,
    now,
    read_json,
    relative,
    resolve,
    sha256_file,
    write_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from moecot_language_tokenizer import exact_text_tokens
from neural_seed_open_vocab import encode_tokens
from kernel_english_protocol import (
    TRAINING_OBJECTIVES,
    TRAINING_VERIFICATION_POLICY,
    compile_training_views,
    kernel_training_contract,
    validate_training_record,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moecot_language_arm_training.json"
ARM_IDS = ("english", "python", "javascript_typescript", "html_css", "rust")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kernel-english", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    cfg = (
        validate_kernel_english_config(config)
        if args.kernel_english
        else validate_config(config)
    )
    if args.kernel_english:
        report = (
            materialize_kernel_english(config, config_path)
            if args.execute
            else inspect_kernel_english(config, config_path)
        )
    else:
        report = materialize(config, config_path) if args.execute else inspect(config, config_path)
    write_json(resolve(args.out or cfg["report"]), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PLANNED"} else 2


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("source_conditioned_pretraining")
    cfg = cfg if isinstance(cfg, dict) else {}
    if cfg.get("policy") != "project_theseus_moecot_source_conditioned_pretraining_v1":
        raise ValueError("unexpected source-conditioned pretraining policy")
    if tuple((cfg.get("rows_by_arm") or {}).keys()) != ARM_IDS:
        raise ValueError("source-conditioned row arm set/order mismatch")
    if int((cfg.get("rows_by_arm") or {}).get("english") or 0) != 0:
        raise ValueError("code-denoising source cannot be assigned to the English arm")
    if not 0.0 < float(cfg.get("deletion_fraction") or 0.0) < 0.5:
        raise ValueError("deletion fraction must be bounded between zero and one half")
    if int(cfg.get("maximum_windows_per_document") or 0) <= 0:
        raise ValueError("maximum windows per document must be positive")
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int(cfg.get(key) or 0):
            raise ValueError(f"source-conditioned no-cheat counter must remain zero: {key}")
    return cfg


def validate_kernel_english_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("kernel_english_training")
    cfg = cfg if isinstance(cfg, dict) else {}
    if cfg.get("policy") != "project_theseus_moecot_kernel_english_stage_v1":
        raise ValueError("unexpected KERC training-stage policy")
    if cfg.get("required") is not True:
        raise ValueError("KERC training stage must remain required for the joint campaign")
    if tuple(cfg.get("objective_order") or ()) != TRAINING_OBJECTIVES:
        raise ValueError("KERC objective order/identity mismatch")
    rows = cfg.get("records_by_split") or {}
    if tuple(rows) != ("private_train", "private_dev", "private_eval"):
        raise ValueError("KERC record split set/order mismatch")
    if any(int(value or 0) <= 0 for value in rows.values()):
        raise ValueError("KERC record floors must be positive for every split")
    if not cfg.get("allowed_licenses"):
        raise ValueError("KERC stage requires an explicit license allowlist")
    if not str(cfg.get("verification_ledger_jsonl") or "").strip():
        raise ValueError("KERC stage requires a separate verification ledger")
    if int(cfg.get("maximum_sequence_tokens") or 0) <= 0:
        raise ValueError("KERC maximum sequence tokens must be positive")
    if not 1 <= int(cfg.get("batch_size") or 0) <= 16:
        raise ValueError("KERC batch size must be bounded")
    for key in (
        "public_training_rows_written",
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
        "template_credit",
        "deterministic_renderer_credit",
        "candidate_generation_credit",
    ):
        if int(cfg.get(key) or 0):
            raise ValueError(f"KERC no-cheat counter must remain zero: {key}")
    return cfg


def inspect_kernel_english(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = validate_kernel_english_config(config)
    manifest_path = resolve(cfg["stage_root"]) / "manifest.json"
    if not manifest_path.is_file():
        return kernel_english_base_report(
            config_path, cfg, "PLANNED", ["kernel_english_stage_not_materialized"]
        )
    payload = read_json(manifest_path)
    gaps = validate_kernel_english_manifest(payload, cfg)
    return {
        **payload,
        "created_utc": now(),
        "mode": "inspection",
        "trigger_state": "RED" if gaps else "GREEN",
        "hard_gaps": gaps,
    }


def materialize_kernel_english(
    config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    cfg = validate_kernel_english_config(config)
    started = time.perf_counter()
    stage_root = resolve(cfg["stage_root"])
    stage_root.mkdir(parents=True, exist_ok=True)
    records_path = resolve(cfg["records_jsonl"])
    ledger_path = resolve(cfg["verification_ledger_jsonl"])
    missing = []
    if not records_path.is_file():
        missing.append("kernel_english_records_missing")
    if not ledger_path.is_file():
        missing.append("kernel_english_verification_ledger_missing")
    if missing:
        report = kernel_english_base_report(
            config_path,
            cfg,
            "RED",
            missing,
        )
        write_json_atomic(stage_root / "manifest.json", report)
        return report

    ledger, ledger_gaps = load_kernel_verification_ledger(ledger_path)
    metadata = read_json(resolve(config["stage_dir"]) / "stage_metadata_v1.json")
    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    selectors = {
        split: BoundedRows(int(count))
        for split, count in cfg["records_by_split"].items()
    }
    rejection_counts: Counter[str] = Counter()
    candidate_count: Counter[str] = Counter()
    for line_number, raw in enumerate(records_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = validate_training_record(json.loads(raw))
        except Exception as exc:
            code = str(getattr(exc, "code", "KERC_RECORD_INVALID"))
            rejection_counts[code] += 1
            continue
        split = str(record["split"])
        candidate_count[split] += 1
        receipt = record["verification_receipt"]
        ledger_receipt = ledger.get(str(receipt["receipt_id"]))
        if ledger_receipt is None:
            rejection_counts["verification_receipt_absent_from_ledger"] += 1
            continue
        if ledger_receipt != receipt:
            rejection_counts["verification_receipt_ledger_mismatch"] += 1
            continue
        if str(record["provenance"]["license_spdx"]).lower() not in {
            str(value).lower() for value in cfg["allowed_licenses"]
        }:
            rejection_counts["license_not_allowed"] += 1
            continue
        selectors[split].add(str(record["record_sha256"]).split(":", 1)[-1], record)

    selected = {split: selector.rows() for split, selector in selectors.items()}
    overlaps = kernel_english_split_overlap(selected)
    gaps = [*ledger_gaps, *overlaps["hard_gaps"]]
    artifacts: dict[str, Any] = {}
    objective_counts: Counter[str] = Counter()
    encoded_length_stats: dict[str, Any] = {}
    all_source_hashes: set[str] = set()
    raw_source_bytes = 0
    for split, records in selected.items():
        wanted = int(cfg["records_by_split"][split])
        if len(records) != wanted:
            gaps.append(f"insufficient_kernel_records:{split}:{len(records)}:{wanted}")
        views: list[dict[str, Any]] = []
        source_lengths: list[int] = []
        target_lengths: list[int] = []
        for record in records:
            all_source_hashes.add(str(record["raw_source_sha256"]))
            raw_source_bytes += len(str(record["source_text"]).encode("utf-8"))
            for view in compile_training_views(record):
                source_body_ids, source_receipt = encode_tokens(
                    exact_text_tokens(view["prompt"]), source_vocab, stream="source"
                )
                trusted_prefix = list(view.get("trusted_source_prefix_tokens") or [])
                if len(trusted_prefix) != 1 or trusted_prefix[0] not in source_vocab:
                    gaps.append(f"kernel_view_trusted_prefix_invalid:{view['row_id']}")
                    continue
                source_ids = [int(source_vocab[trusted_prefix[0]]), *source_body_ids]
                target_ids, target_receipt = encode_tokens(
                    exact_text_tokens(view["target"]), target_vocab, stream="target"
                )
                if int(source_receipt.get("unknown_token_count") or 0) or int(
                    target_receipt.get("unknown_token_count") or 0
                ):
                    gaps.append(f"kernel_view_unrepresentable:{view['row_id']}")
                    continue
                sequence_tokens = len(source_ids) + len(target_ids) + 4
                if sequence_tokens > int(cfg["maximum_sequence_tokens"]):
                    gaps.append(
                        f"kernel_view_requires_truncation:{view['row_id']}:{sequence_tokens}"
                    )
                    continue
                source_lengths.append(len(source_ids))
                target_lengths.append(len(target_ids))
                objective_counts[str(view["objective"])] += 1
                views.append(view)
        path = stage_root / f"{split}.jsonl"
        write_jsonl_atomic(path, views)
        artifacts[f"english:{split}"] = {
            "path": relative(path),
            "sha256": sha256_file(path),
            "row_count": len(views),
            "unique_record_count": len(records),
            "bytes": path.stat().st_size,
        }
        encoded_length_stats[split] = {
            "maximum_source_tokens": max(source_lengths or [0]),
            "maximum_target_tokens": max(target_lengths or [0]),
            "maximum_sequence_tokens": max(
                (source + target + 4 for source, target in zip(source_lengths, target_lengths)),
                default=0,
            ),
        }

    report = {
        "policy": cfg["policy"],
        "created_utc": now(),
        "mode": "materialized",
        "trigger_state": "RED" if gaps else "GREEN",
        "config": relative(config_path),
        "contract_sha256": kernel_english_stage_contract_sha256(cfg),
        "learned_pipeline_contract": kernel_training_contract(),
        "required_records_by_split": dict(cfg["records_by_split"]),
        "verification_ledger_required": True,
        "source": {
            "path": relative(records_path),
            "sha256": sha256_file(records_path),
            "license_policy": "row_level_explicit_allowlist",
        },
        "verification_ledger": {
            "path": relative(ledger_path),
            "sha256": sha256_file(ledger_path),
            "receipt_count": len(ledger),
            "producer_separate_from_training_rows": True,
        },
        "artifacts": artifacts,
        "candidate_record_count_by_split": dict(candidate_count),
        "selected_record_count_by_split": {
            split: len(records) for split, records in selected.items()
        },
        "compiled_view_count_by_objective": dict(objective_counts),
        "unique_raw_source_count": len(all_source_hashes),
        "unique_raw_source_bytes": raw_source_bytes,
        "derived_view_unique_data_credit": 0,
        "derived_view_optimizer_exposure_count": sum(objective_counts.values()),
        "split_overlap_audit": overlaps,
        "encoded_length_stats": encoded_length_stats,
        "rejection_counts": dict(rejection_counts),
        "failure_behavior": "reject_without_template_literal_tool_or_router_fallback",
        "score_semantics": "KERC learned-objective data readiness; not learned capability",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "hard_gaps": sorted(set(gaps)),
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "deterministic_renderer_credit": 0,
        "candidate_generation_credit": 0,
    }
    write_json_atomic(stage_root / "manifest.json", report)
    return report


def kernel_english_split_overlap(
    selected: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    groups: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    for split, records in selected.items():
        groups[split] = {str(row["provenance"]["source_group"]) for row in records}
        sources[split] = {str(row["raw_source_sha256"]) for row in records}
    group_overlap = 0
    source_overlap = 0
    for left_index, left in enumerate(selected):
        for right in tuple(selected)[left_index + 1 :]:
            group_overlap += len(groups[left] & groups[right])
            source_overlap += len(sources[left] & sources[right])
    gaps = []
    if group_overlap:
        gaps.append(f"kernel_source_group_cross_split_overlap:{group_overlap}")
    if source_overlap:
        gaps.append(f"kernel_raw_source_cross_split_overlap:{source_overlap}")
    return {
        "source_group_overlap_count": group_overlap,
        "raw_source_overlap_count": source_overlap,
        "content_bound_disjoint": not gaps,
        "hard_gaps": gaps,
    }


def load_kernel_verification_ledger(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    receipts: dict[str, dict[str, Any]] = {}
    gaps: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            gaps.append(f"kernel_verification_ledger_json_invalid:{line_number}")
            continue
        receipt_id = str(row.get("receipt_id") or "") if isinstance(row, dict) else ""
        if not receipt_id:
            gaps.append(f"kernel_verification_ledger_receipt_id_missing:{line_number}")
            continue
        if receipt_id in receipts:
            gaps.append(f"kernel_verification_ledger_receipt_duplicate:{receipt_id}")
            continue
        if row.get("policy") != TRAINING_VERIFICATION_POLICY:
            gaps.append(f"kernel_verification_ledger_policy_invalid:{receipt_id}")
            continue
        if row.get("accepted") is not True:
            gaps.append(f"kernel_verification_ledger_unaccepted:{receipt_id}")
            continue
        receipts[receipt_id] = row
    return receipts, sorted(set(gaps))


def validate_kernel_english_manifest(
    payload: dict[str, Any], cfg: dict[str, Any]
) -> list[str]:
    gaps: list[str] = []
    if payload.get("policy") != cfg["policy"]:
        gaps.append("kernel_stage_policy_mismatch")
    if payload.get("contract_sha256") != kernel_english_stage_contract_sha256(cfg):
        gaps.append("kernel_stage_contract_identity_mismatch")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    objective_count = len(TRAINING_OBJECTIVES)
    for split, record_count in cfg["records_by_split"].items():
        key = f"english:{split}"
        artifact = artifacts.get(key) if isinstance(artifacts.get(key), dict) else {}
        path = resolve(str(artifact.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(artifact.get("sha256") or ""):
            gaps.append(f"kernel_stage_artifact_identity_mismatch:{key}")
        if int(artifact.get("row_count") or 0) != int(record_count) * objective_count:
            gaps.append(f"kernel_stage_view_count_mismatch:{key}")
        if int(artifact.get("unique_record_count") or 0) != int(record_count):
            gaps.append(f"kernel_stage_record_count_mismatch:{key}")
    overlap = payload.get("split_overlap_audit") or {}
    if not bool(overlap.get("content_bound_disjoint")):
        gaps.append("kernel_stage_split_overlap")
    if int(payload.get("derived_view_unique_data_credit") or 0):
        gaps.append("kernel_stage_derived_view_unique_credit_nonzero")
    ledger = payload.get("verification_ledger") or {}
    ledger_path = resolve(str(ledger.get("path") or ""))
    if (
        not ledger_path.is_file()
        or sha256_file(ledger_path) != str(ledger.get("sha256") or "")
    ):
        gaps.append("kernel_stage_verification_ledger_identity_mismatch")
    if ledger.get("producer_separate_from_training_rows") is not True:
        gaps.append("kernel_stage_verification_ledger_not_independent")
    for key in (
        "public_training_rows_written",
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
        "template_credit",
        "deterministic_renderer_credit",
        "candidate_generation_credit",
    ):
        if int(payload.get(key) or 0):
            gaps.append(f"kernel_stage_nonzero_boundary:{key}")
    return sorted(set(gaps))


def kernel_english_stage_contract_sha256(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def kernel_english_base_report(
    config_path: Path,
    cfg: dict[str, Any],
    state: str,
    gaps: list[str],
) -> dict[str, Any]:
    return {
        "policy": cfg["policy"],
        "created_utc": now(),
        "mode": "inspection",
        "trigger_state": state,
        "config": relative(config_path),
        "contract_sha256": kernel_english_stage_contract_sha256(cfg),
        "learned_pipeline_contract": kernel_training_contract(),
        "required_records_by_split": dict(cfg["records_by_split"]),
        "verification_ledger_required": True,
        "hard_gaps": gaps,
        "score_semantics": "KERC learned-objective data readiness; not learned capability",
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "deterministic_renderer_credit": 0,
        "candidate_generation_credit": 0,
    }


def inspect(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = validate_config(config)
    manifest_path = resolve(cfg["stage_root"]) / "manifest.json"
    if not manifest_path.is_file():
        return base_report(config_path, cfg, "PLANNED", ["stage_not_materialized"])
    payload = read_json(manifest_path)
    gaps = validate_manifest(payload, cfg)
    return {
        **payload,
        "created_utc": now(),
        "mode": "inspection",
        "trigger_state": "RED" if gaps else "GREEN",
        "hard_gaps": gaps,
    }


def materialize(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = validate_config(config)
    started = time.perf_counter()
    stage_root = resolve(cfg["stage_root"])
    stage_root.mkdir(parents=True, exist_ok=True)
    metadata = read_json(resolve(config["stage_dir"]) / "stage_metadata_v1.json")
    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    supervision_targets = supervision_target_hashes(config)
    selectors = {
        arm: BoundedRows(int(count))
        for arm, count in cfg["rows_by_arm"].items()
        if int(count) > 0
    }
    language_to_arm = {
        language: arm
        for arm, languages in (cfg.get("arm_languages") or {}).items()
        for language in languages
    }
    source_path = resolve(cfg["source_jsonl"])
    rejections: Counter[str] = Counter()
    candidate_count: Counter[str] = Counter()
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            arm = language_to_arm.get(str(source.get("language") or "").lower())
            if arm not in selectors:
                continue
            reason = source_rejection(source, cfg)
            if reason:
                rejections[reason] += 1
                continue
            for row in denoising_rows(source, arm, cfg, source_vocab, target_vocab):
                candidate_count[arm] += 1
                if row["target_sha256"] in supervision_targets:
                    rejections["supervision_target_overlap"] += 1
                    continue
                selectors[arm].add(row["selection_sha256"], row)

    artifacts: dict[str, Any] = {}
    copy_coverage: dict[str, Any] = {}
    gaps: list[str] = []
    for arm, selector in selectors.items():
        rows = selector.rows()
        wanted = int(cfg["rows_by_arm"][arm])
        if len(rows) != wanted:
            gaps.append(f"insufficient_rows:{arm}:{len(rows)}:{wanted}")
        path = stage_root / f"{arm}.jsonl"
        write_jsonl_atomic(path, rows)
        artifacts[arm] = {
            "path": relative(path),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "bytes": path.stat().st_size,
        }
        fractions = [float(row["target_token_copy_fraction"]) for row in rows]
        copy_coverage[arm] = {
            "mean_target_token_copy_fraction": round(
                sum(fractions) / max(1, len(fractions)), 8
            ),
            "minimum_target_token_copy_fraction": round(min(fractions or [0.0]), 8),
        }
    report = {
        "policy": cfg["policy"],
        "created_utc": now(),
        "mode": "materialized",
        "trigger_state": "RED" if gaps else "GREEN",
        "config": relative(config_path),
        "contract_sha256": contract_sha256(cfg),
        "source": {
            "path": relative(source_path),
            "sha256": sha256_file(source_path),
            "license_policy": "row_level_permissive_allowlist",
        },
        "artifacts": artifacts,
        "candidate_count_by_arm": dict(candidate_count),
        "copy_coverage_by_arm": copy_coverage,
        "rejection_counts": dict(rejections),
        "supervision_target_overlap_count": int(rejections["supervision_target_overlap"]),
        "corruption": {
            "mode": "deterministic_span_deletion_reconstruction",
            "deletion_fraction": float(cfg["deletion_fraction"]),
            "maximum_spans": int(cfg["maximum_deletion_spans"]),
            "seed": int(cfg["seed"]),
        },
        "generator_visible_fields": ["prompt"],
        "evaluator_only_fields": ["target", "target_sha256", "source_identity"],
        "score_semantics": "licensed source-conditioned objective readiness; not edit capability",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "hard_gaps": gaps,
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    write_json_atomic(stage_root / "manifest.json", report)
    return report


def denoising_rows(
    source: dict[str, Any],
    arm: str,
    cfg: dict[str, Any],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
) -> list[dict[str, Any]]:
    text = str(source.get("text") or "")
    logical = exact_text_tokens(text)
    minimum = int(cfg["minimum_target_logical_tokens"])
    maximum = int(cfg["maximum_target_logical_tokens"])
    if len(logical) < minimum:
        return []
    source_identity = str(source.get("text_sha256") or hashlib.sha256(text.encode()).hexdigest())
    starts = list(range(0, len(logical) - minimum + 1, maximum))
    ranked_starts = sorted(
        starts,
        key=lambda start: hashlib.sha256(f"{source_identity}:{start}:{cfg['seed']}".encode()).hexdigest(),
    )[: int(cfg["maximum_windows_per_document"])]
    rows = []
    for start in ranked_starts:
        target_tokens = logical[start : start + maximum]
        if len(target_tokens) < minimum:
            continue
        corruption_identity = hashlib.sha256(
            f"{source_identity}:{start}:{cfg['seed']}".encode()
        ).hexdigest()
        damaged_tokens = delete_spans(target_tokens, cfg, corruption_identity)
        target = "".join(target_tokens)
        damaged = "".join(damaged_tokens)
        if not target.strip() or damaged == target:
            continue
        language = str(source.get("language") or arm)
        prompt = (
            f"Reconstruct the complete original {language} excerpt from this damaged excerpt. "
            "Return only the original excerpt.\n\n"
            f"Damaged excerpt:\n{damaged}"
        )
        source_ids, source_receipt = encode_tokens(
            exact_text_tokens(prompt), source_vocab, stream="source"
        )
        target_ids, target_receipt = encode_tokens(
            exact_text_tokens(target), target_vocab, stream="target"
        )
        if int(source_receipt.get("unknown_token_count") or 0) or int(
            target_receipt.get("unknown_token_count") or 0
        ):
            continue
        if len(source_ids) > int(cfg["maximum_source_encoded_tokens"]) or len(
            target_ids
        ) > int(cfg["maximum_target_encoded_tokens"]):
            continue
        source_token_set = set(exact_text_tokens(prompt))
        copy_fraction = sum(token in source_token_set for token in exact_text_tokens(target)) / max(
            1, len(exact_text_tokens(target))
        )
        digest = hashlib.sha256(
            f"{arm}:{source_identity}:{start}:{corruption_identity}".encode()
        ).hexdigest()
        rows.append(
            {
                "row_id": f"moecot-denoise-{digest[:20]}",
                "split": "private_train",
                "arm_id": arm,
                "objective": "source_conditioned_span_deletion_reconstruction_v1",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "target": target,
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                "target_token_copy_fraction": round(copy_fraction, 8),
                "selection_sha256": digest,
                "source_identity": {
                    "repo": source.get("repo"),
                    "path": source.get("path"),
                    "text_sha256": source_identity,
                    "window_start": start,
                    "license_spdx": source.get("license_spdx"),
                },
                "public_benchmark": False,
                "public_tests_included": False,
                "public_benchmark_solutions_included": False,
                "external_inference": False,
            }
        )
    return rows


def delete_spans(tokens: list[str], cfg: dict[str, Any], identity: str) -> list[str]:
    rng = random.Random(int(identity[:16], 16))
    delete_count = max(1, round(len(tokens) * float(cfg["deletion_fraction"])))
    spans = min(int(cfg["maximum_deletion_spans"]), delete_count)
    removed: set[int] = set()
    remaining = delete_count
    for span_index in range(spans):
        width = max(1, remaining // (spans - span_index))
        start = rng.randrange(max(1, len(tokens) - width + 1))
        removed.update(range(start, min(len(tokens), start + width)))
        remaining = max(0, delete_count - len(removed))
    while len(removed) < delete_count:
        removed.add(rng.randrange(len(tokens)))
    return [token for index, token in enumerate(tokens) if index not in removed]


def source_rejection(source: dict[str, Any], cfg: dict[str, Any]) -> str:
    if source.get("public_benchmark") is not False:
        return "public_benchmark_state_not_false"
    if source.get("public_tests_included") is not False:
        return "public_tests_present"
    if source.get("public_benchmark_solutions_included") is not False:
        return "public_solutions_present"
    if str(source.get("license_spdx") or "").lower() not in {
        str(value).lower() for value in cfg["allowed_licenses"]
    }:
        return "license_not_allowed"
    if not str(source.get("text") or "").strip():
        return "empty_text"
    return ""


def supervision_target_hashes(config: dict[str, Any]) -> set[str]:
    root = resolve(config["supervision"]["stage_root"])
    hashes: set[str] = set()
    for path in sorted(root.glob("private_*/*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                hashes.add(str(row.get("target_sha256") or ""))
    return hashes


def contract_sha256(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_manifest(payload: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    gaps = []
    if payload.get("policy") != cfg["policy"]:
        gaps.append("policy_mismatch")
    if payload.get("contract_sha256") != contract_sha256(cfg):
        gaps.append("contract_identity_mismatch")
    for arm, wanted in cfg["rows_by_arm"].items():
        if int(wanted) <= 0:
            continue
        artifact = (payload.get("artifacts") or {}).get(arm) or {}
        path = resolve(str(artifact.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(artifact.get("sha256") or ""):
            gaps.append(f"artifact_identity_mismatch:{arm}")
        if int(artifact.get("row_count") or 0) != int(wanted):
            gaps.append(f"row_count_mismatch:{arm}")
    for key in ("public_training_rows_written", "public_benchmark_payload_count", "external_inference_calls", "fallback_return_count"):
        if int(payload.get(key) or 0):
            gaps.append(f"nonzero_boundary:{key}")
    return gaps


def base_report(
    config_path: Path, cfg: dict[str, Any], state: str, gaps: list[str]
) -> dict[str, Any]:
    return {
        "policy": cfg["policy"],
        "created_utc": now(),
        "mode": "inspection",
        "trigger_state": state,
        "config": relative(config_path),
        "contract_sha256": contract_sha256(cfg),
        "hard_gaps": gaps,
        "score_semantics": "licensed source-conditioned objective readiness; not capability",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
