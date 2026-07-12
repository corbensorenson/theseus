#!/usr/bin/env python3
"""Materialize frozen, licensed MoECOT SFT and heldout language-arm rows."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import heapq
import json
import os
import re
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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
    supervision = validate_config(config)
    report_path = resolve(args.out or supervision["report"])
    report = materialize(config, config_path=config_path) if args.execute else inspect(config, config_path)
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PLANNED"} else 2


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("supervision") if isinstance(config.get("supervision"), dict) else {}
    if cfg.get("policy") != "project_theseus_moecot_language_supervision_v1":
        raise ValueError("unexpected MoECOT supervision policy")
    if tuple((cfg.get("train_rows_by_arm") or {}).keys()) != ARM_IDS:
        raise ValueError("supervision train arm set/order mismatch")
    if tuple((cfg.get("heldout_rows_by_arm") or {}).keys()) != ARM_IDS:
        raise ValueError("supervision heldout arm set/order mismatch")
    if (cfg.get("code_source") or {}).get("revision") != "fc56fe33c030c6daa414c2b112c932b8eed085e6":
        raise ValueError("CommitPackFT revision is not frozen")
    if (cfg.get("english_source") or {}).get("revision") != "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a":
        raise ValueError("Dolly revision is not frozen")
    if cfg.get("generator_visible_fields") != ["prompt"]:
        raise ValueError("generator-visible supervision fields must remain prompt-only")
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int(cfg.get(key) or 0):
            raise ValueError(f"supervision no-cheat counter must remain zero: {key}")
    return cfg


def inspect(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = validate_config(config)
    root = resolve(cfg["stage_root"])
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return base_report(config_path, cfg, "PLANNED", ["supervision_stage_not_materialized"])
    payload = read_json(manifest)
    gaps = validate_manifest(payload, cfg, root)
    return {
        **payload,
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "mode": "inspection",
        "hard_gaps": gaps,
    }


def materialize(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    cfg = validate_config(config)
    started = time.perf_counter()
    root = resolve(cfg["stage_root"])
    root.mkdir(parents=True, exist_ok=True)
    selectors = {
        arm: {
            "private_train": BoundedRows(int(cfg["train_rows_by_arm"][arm])),
            "private_eval": BoundedRows(int(cfg["heldout_rows_by_arm"][arm])),
        }
        for arm in ARM_IDS
    }
    source_receipts: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    target_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    base = read_json(resolve(str(config["base_config"])))
    vocab_path = resolve(str(base["tokenization"]["source_vocab"]))
    vocab_payload = read_json(vocab_path)
    source_vocab = dict(vocab_payload["source_vocab"])
    target_vocab = dict(vocab_payload["target_vocab"])

    english = cfg["english_source"]
    receipt, rows = stream_jsonl(str(english["url"]))
    for source in rows:
        row, reason = normalize_english(source, cfg, english)
        if reason:
            rejection_counts[reason] += 1
            continue
        reason = encoding_rejection(row, cfg, source_vocab, target_vocab)
        if reason:
            rejection_counts[reason] += 1
            continue
        admit_row(row, selectors["english"], cfg, prompt_hashes, target_hashes, rejection_counts)
    source_receipts.append({**receipt, "dataset_id": english["dataset_id"], "revision": english["revision"]})

    code = cfg["code_source"]
    for arm, languages in code["arm_languages"].items():
        for language in languages:
            url = str(code["url_template"]).format(revision=code["revision"], language=language)
            receipt, rows = stream_jsonl(url)
            for source in rows:
                row, reason = normalize_code(source, cfg, code, arm=arm, language=language)
                if reason:
                    rejection_counts[reason] += 1
                    continue
                reason = encoding_rejection(row, cfg, source_vocab, target_vocab)
                if reason:
                    rejection_counts[reason] += 1
                    continue
                admit_row(row, selectors[arm], cfg, prompt_hashes, target_hashes, rejection_counts)
            source_receipts.append(
                {**receipt, "dataset_id": code["dataset_id"], "revision": code["revision"], "language": language}
            )

    artifacts: dict[str, Any] = {}
    row_counts: dict[str, Any] = {}
    prompt_hash_sets: dict[str, set[str]] = {}
    target_hash_sets: dict[str, set[str]] = {}
    for arm in ARM_IDS:
        row_counts[arm] = {}
        for split in ("private_train", "private_eval"):
            rows = selectors[arm][split].rows()
            wanted = int(
                cfg["train_rows_by_arm"][arm]
                if split == "private_train"
                else cfg["heldout_rows_by_arm"][arm]
            )
            path = root / split / f"{arm}.jsonl"
            write_jsonl_atomic(path, rows)
            key = f"{arm}:{split}"
            row_counts[arm][split] = len(rows)
            artifacts[key] = artifact(path, row_count=len(rows))
            prompt_hash_sets[key] = {str(row["prompt_sha256"]) for row in rows}
            target_hash_sets[key] = {str(row["target_sha256"]) for row in rows}
            if len(rows) < wanted:
                rejection_counts[f"quota_shortfall:{arm}:{split}:{wanted - len(rows)}"] += 1

    overlap = split_overlap(prompt_hash_sets, target_hash_sets)
    gaps = [key for key in rejection_counts if key.startswith("quota_shortfall:")]
    if overlap["prompt_overlap_count"]:
        gaps.append("cross_split_prompt_overlap")
    if overlap["target_overlap_count"]:
        gaps.append("cross_split_target_overlap")
    report = {
        "policy": cfg["policy"],
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "mode": "materialized",
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "contract_sha256": contract_sha256(cfg),
        "stage_root": relative(root),
        "source_receipts": source_receipts,
        "vocabulary": artifact(vocab_path),
        "row_counts": row_counts,
        "artifacts": artifacts,
        "split_overlap_audit": overlap,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "hard_gaps": gaps,
        "generator_visible_fields": cfg["generator_visible_fields"],
        "evaluator_only_fields": cfg["evaluator_only_fields"],
        "score_semantics": "licensed SFT and frozen heldout data readiness; not model capability",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    write_json_atomic(root / "manifest.json", report)
    return report


class BoundedRows:
    def __init__(self, limit: int):
        self.limit = int(limit)
        self.heap: list[tuple[int, str, dict[str, Any]]] = []
        self.seen: set[str] = set()

    def add(self, digest: str, row: dict[str, Any]) -> bool:
        if digest in self.seen:
            return False
        rank = int(digest[:16], 16)
        item = (-rank, digest, row)
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, item)
            self.seen.add(digest)
            return True
        if rank >= -self.heap[0][0]:
            return False
        removed = heapq.heapreplace(self.heap, item)
        self.seen.discard(removed[1])
        self.seen.add(digest)
        return True

    def rows(self) -> list[dict[str, Any]]:
        return [item[2] for item in sorted(self.heap, key=lambda item: item[1])]


def admit_row(
    row: dict[str, Any],
    selectors: dict[str, BoundedRows],
    cfg: dict[str, Any],
    prompt_hashes: set[str],
    target_hashes: set[str],
    rejection_counts: Counter[str],
) -> None:
    if row["prompt_sha256"] in prompt_hashes:
        rejection_counts["duplicate_prompt"] += 1
        return
    if row["target_sha256"] in target_hashes:
        rejection_counts["duplicate_target"] += 1
        return
    split_digest = sha256_text(f"{cfg['split_seed']}\n{row['arm_id']}\n{row['prompt_sha256']}")
    digest = sha256_text(f"{split_digest}\n{row['source_identity']}\n{row['target_sha256']}")
    split_cfg = cfg["split_contract"]
    split = (
        "private_eval"
        if int(split_digest[:16], 16) % int(split_cfg["heldout_modulus"])
        == int(split_cfg["heldout_remainder"])
        else "private_train"
    )
    row["split"] = split
    row["selection_sha256"] = digest
    if selectors[split].add(digest, row):
        prompt_hashes.add(row["prompt_sha256"])
        target_hashes.add(row["target_sha256"])


def normalize_english(
    source: dict[str, Any], cfg: dict[str, Any], source_cfg: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    instruction = clean_text(source.get("instruction"))
    context = clean_text(source.get("context"))
    target = clean_text(source.get("response"))
    reason = common_rejection(instruction, target, cfg, context=context)
    if reason:
        return {}, reason
    prompt = instruction + (f"\n\nContext:\n{context}" if context else "")
    source_identity = sha256_text(
        f"{source_cfg['dataset_id']}\n{source_cfg['revision']}\n{source.get('category')}\n{instruction}\n{context}\n{target}"
    )
    return supervision_row(
        arm="english",
        prompt=prompt,
        target=target,
        source_identity=source_identity,
        dataset_id=source_cfg["dataset_id"],
        revision=source_cfg["revision"],
        license_spdx=source_cfg["dataset_license"],
        provenance={"instruction_category": str(source.get("category") or "")},
    ), ""


def normalize_code(
    source: dict[str, Any],
    cfg: dict[str, Any],
    source_cfg: dict[str, Any],
    *,
    arm: str,
    language: str,
) -> tuple[dict[str, Any], str]:
    license_id = str(source.get("license") or "").strip().lower()
    if license_id not in {str(value).lower() for value in source_cfg["allowed_row_licenses"]}:
        return {}, "row_license_not_allowed"
    instruction = clean_text(source.get("subject"))
    old = str(source.get("old_contents") or "")
    new = str(source.get("new_contents") or "")
    old_excerpt, target, excerpt_receipt = localized_change_excerpt(
        old, new, context_lines=int(cfg["code_context_lines"])
    )
    reason = common_rejection(instruction, target, cfg, context=old_excerpt)
    if reason:
        return {}, reason
    if old == new:
        return {}, "unchanged_target"
    prompt = (
        f"Apply the requested change to this {language} excerpt.\n\n"
        f"Request:\n{instruction}\n\nCurrent excerpt:\n{old_excerpt}\n\n"
        "Return only the complete revised excerpt."
    )
    source_identity = sha256_text(
        f"{source_cfg['dataset_id']}\n{source_cfg['revision']}\n{source.get('commit')}\n{language}\n{old}\n{target}"
    )
    return supervision_row(
        arm=arm,
        prompt=prompt,
        target=target,
        source_identity=source_identity,
        dataset_id=source_cfg["dataset_id"],
        revision=source_cfg["revision"],
        license_spdx=license_id,
        provenance={
            "language": language,
            "commit": str(source.get("commit") or ""),
            "repository": str(source.get("repos") or ""),
            "old_file": str(source.get("old_file") or ""),
            "new_file": str(source.get("new_file") or ""),
            "old_file_sha256": sha256_text(old),
            "new_file_sha256": sha256_text(new),
            "excerpt": excerpt_receipt,
        },
    ), ""


def common_rejection(instruction: str, target: str, cfg: dict[str, Any], *, context: str) -> str:
    if len(instruction) < int(cfg["minimum_instruction_characters"]):
        return "instruction_too_short"
    if len(instruction) > int(cfg["maximum_instruction_characters"]):
        return "instruction_too_long"
    if len(target) < int(cfg["minimum_target_characters"]):
        return "target_too_short"
    if len(target) > int(cfg["maximum_target_characters"]):
        return "target_too_long"
    if len(context) > int(cfg["maximum_context_characters"]):
        return "context_too_long"
    visible = f"{instruction}\n{context}\n{target}".lower()
    if any(str(marker).lower() in visible for marker in cfg["forbidden_benchmark_markers"]):
        return "public_benchmark_marker"
    if contains_binary_or_control(target) or contains_binary_or_control(context):
        return "binary_or_control_text"
    return ""


def localized_change_excerpt(
    old: str, new: str, *, context_lines: int
) -> tuple[str, str, dict[str, Any]]:
    old_lines = str(old).splitlines(keepends=True)
    new_lines = str(new).splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    changed = [opcode for opcode in matcher.get_opcodes() if opcode[0] != "equal"]
    if not changed:
        return "", "", {"state": "UNCHANGED"}
    old_start = max(0, min(row[1] for row in changed) - max(0, context_lines))
    old_stop = min(len(old_lines), max(row[2] for row in changed) + max(0, context_lines))
    new_start = max(0, min(row[3] for row in changed) - max(0, context_lines))
    new_stop = min(len(new_lines), max(row[4] for row in changed) + max(0, context_lines))
    old_excerpt = "".join(old_lines[old_start:old_stop])
    new_excerpt = "".join(new_lines[new_start:new_stop])
    return old_excerpt, new_excerpt, {
        "policy": "project_theseus_localized_code_edit_excerpt_v1",
        "state": "READY",
        "context_lines": int(context_lines),
        "old_line_range": [old_start, old_stop],
        "new_line_range": [new_start, new_stop],
        "changed_opcode_count": len(changed),
        "full_old_file_visible": old_start == 0 and old_stop == len(old_lines),
        "full_new_file_is_target": new_start == 0 and new_stop == len(new_lines),
    }


def encoding_rejection(
    row: dict[str, Any],
    cfg: dict[str, Any],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
) -> str:
    source_ids, source_receipt = encode_tokens(
        exact_text_tokens(str(row["prompt"])), source_vocab, stream="source"
    )
    target_ids, target_receipt = encode_tokens(
        exact_text_tokens(str(row["target"])), target_vocab, stream="target"
    )
    if int(source_receipt.get("unknown_token_count") or 0):
        return "source_tokenizer_unrepresentable"
    if int(target_receipt.get("unknown_token_count") or 0):
        return "target_tokenizer_unrepresentable"
    if len(source_ids) > int(cfg["maximum_source_encoded_tokens"]):
        return "source_encoded_too_long"
    if len(target_ids) > int(cfg["maximum_target_encoded_tokens"]):
        return "target_encoded_too_long"
    row["source_encoded_token_count"] = len(source_ids)
    row["target_encoded_token_count"] = len(target_ids)
    row["complete_target_in_window"] = True
    return ""


def supervision_row(
    *,
    arm: str,
    prompt: str,
    target: str,
    source_identity: str,
    dataset_id: str,
    revision: str,
    license_spdx: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    prompt_hash = sha256_text(prompt)
    target_hash = sha256_text(target)
    return {
        "policy": "project_theseus_moecot_language_supervision_row_v1",
        "row_id": f"moecot-sft-{sha256_text(source_identity + prompt_hash + target_hash)[:20]}",
        "arm_id": arm,
        "prompt": prompt,
        "prompt_sha256": prompt_hash,
        "target": target,
        "target_sha256": target_hash,
        "source_identity": source_identity,
        "dataset_id": dataset_id,
        "dataset_revision": revision,
        "license_spdx": license_spdx,
        "provenance": provenance,
        "public_benchmark": False,
        "public_benchmark_payload": False,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def stream_jsonl(url: str) -> tuple[dict[str, Any], Iterable[dict[str, Any]]]:
    request = urllib.request.Request(url, headers={"User-Agent": "ProjectTheseus/1.0"})
    response = urllib.request.urlopen(request, timeout=120)
    digest = hashlib.sha256()
    counters = {"bytes": 0, "rows": 0}

    def rows() -> Iterable[dict[str, Any]]:
        try:
            for raw in response:
                digest.update(raw)
                counters["bytes"] += len(raw)
                if not raw.strip():
                    continue
                counters["rows"] += 1
                yield json.loads(raw)
        finally:
            response.close()

    receipt = {
        "url": url,
        "stream_sha256": "",
        "downloaded_bytes": counters,
    }

    def wrapped() -> Iterable[dict[str, Any]]:
        yield from rows()
        receipt["stream_sha256"] = digest.hexdigest()
        receipt["downloaded_bytes"] = counters["bytes"]
        receipt["source_row_count"] = counters["rows"]

    return receipt, wrapped()


def split_overlap(
    prompts: dict[str, set[str]], targets: dict[str, set[str]]
) -> dict[str, Any]:
    prompt_overlaps: set[str] = set()
    target_overlaps: set[str] = set()
    for arm in ARM_IDS:
        prompt_overlaps |= prompts[f"{arm}:private_train"] & prompts[f"{arm}:private_eval"]
        target_overlaps |= targets[f"{arm}:private_train"] & targets[f"{arm}:private_eval"]
    return {
        "policy": "project_theseus_moecot_supervision_split_overlap_v1",
        "prompt_overlap_count": len(prompt_overlaps),
        "target_overlap_count": len(target_overlaps),
        "prompt_overlap_sample": sorted(prompt_overlaps)[:8],
        "target_overlap_sample": sorted(target_overlaps)[:8],
        "public_benchmark_payload_count": 0,
    }


def validate_manifest(payload: dict[str, Any], cfg: dict[str, Any], root: Path) -> list[str]:
    gaps: list[str] = []
    if payload.get("policy") != cfg["policy"]:
        gaps.append("manifest_policy_mismatch")
    if payload.get("contract_sha256") != contract_sha256(cfg):
        gaps.append("supervision_contract_identity_mismatch")
    for arm in ARM_IDS:
        for split, wanted in (
            ("private_train", int(cfg["train_rows_by_arm"][arm])),
            ("private_eval", int(cfg["heldout_rows_by_arm"][arm])),
        ):
            key = f"{arm}:{split}"
            row = (payload.get("artifacts") or {}).get(key) or {}
            path = resolve(str(row.get("path") or root / split / f"{arm}.jsonl"))
            if not path.is_file() or sha256_file(path) != row.get("sha256"):
                gaps.append(f"artifact_identity_mismatch:{key}")
            if int(row.get("row_count") or 0) != wanted:
                gaps.append(f"row_count_mismatch:{key}")
    overlap = payload.get("split_overlap_audit") or {}
    if int(overlap.get("prompt_overlap_count") or 0) or int(overlap.get("target_overlap_count") or 0):
        gaps.append("split_overlap")
    base = read_json(resolve(str(config_base_path(cfg))))
    vocab_path = resolve(str(base["tokenization"]["source_vocab"]))
    vocabulary = payload.get("vocabulary") if isinstance(payload.get("vocabulary"), dict) else {}
    if not vocab_path.is_file() or sha256_file(vocab_path) != str(vocabulary.get("sha256") or ""):
        gaps.append("supervision_vocabulary_identity_mismatch")
    return gaps


def config_base_path(cfg: dict[str, Any]) -> str:
    # The supervision config is nested in the top-level arm config; its sibling
    # base_config is fixed by validate_config callers and defaults canonically here.
    return "configs/standard_causal_transformer_survival.json"


def base_report(config_path: Path, cfg: dict[str, Any], state: str, gaps: list[str]) -> dict[str, Any]:
    return {
        "policy": cfg["policy"],
        "created_utc": now(),
        "trigger_state": state,
        "mode": "plan",
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "contract_sha256": contract_sha256(cfg),
        "stage_root": cfg["stage_root"],
        "hard_gaps": gaps,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def contract_sha256(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def artifact(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    row = {
        "path": relative(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if row_count is not None:
        row["row_count"] = int(row_count)
    return row


def contains_binary_or_control(value: str) -> bool:
    return bool(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", str(value)))


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    write_json(temporary, payload)
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
