#!/usr/bin/env python3
"""Materialize licensed source-conditioned denoising rows for MoECOT arms."""

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moecot_language_arm_training.json"
ARM_IDS = ("english", "python", "javascript_typescript", "html_css", "rust")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    cfg = validate_config(config)
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
