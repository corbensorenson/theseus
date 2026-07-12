"""Canonical, content-bound pretraining stage materialization.

The corpus is indexed by immutable JSONL byte ranges, selected by deterministic
category budgets, and written as fixed-width disk-backed arrays. No corpus-sized
Python list or MLX tensor is constructed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from governed_conversation_stream import ConversationDeduper


CATEGORY_ORDER = (
    "english_conversation_instruction",
    "english_broad",
    "python",
    "javascript_typescript",
    "html_css",
    "rust",
)


def category_targets(contract: dict[str, Any]) -> dict[str, int]:
    domains = contract.get("domain_minimum_positions") or {}
    subsets = contract.get("subset_minimum_positions") or {}
    languages = contract.get("code_language_minimum_positions") or {}
    conversation = int(subsets.get("english_conversation_instruction") or 0)
    english_total = int(domains.get("english_natural_language_total") or 0)
    targets = {
        "english_conversation_instruction": conversation,
        "english_broad": english_total - conversation,
        "python": int(languages.get("python") or 0),
        "javascript_typescript": int(languages.get("javascript_typescript") or 0),
        "html_css": int(languages.get("html_css") or 0),
        "rust": int(languages.get("rust") or 0),
    }
    if min(targets.values()) <= 0:
        raise ValueError("canonical category minima must all be positive")
    if sum(targets[key] for key in CATEGORY_ORDER[2:]) != int(domains.get("code_total") or 0):
        raise ValueError("canonical code language minima must partition code total")
    tail = int(domains.get("flexible_tail_reserve") or 0)
    share, remainder = divmod(tail, len(CATEGORY_ORDER))
    for index, category in enumerate(CATEGORY_ORDER):
        targets[category] += share + (1 if index < remainder else 0)
    required = int(contract.get("required_unique_positions") or 0)
    if sum(targets.values()) != required:
        raise ValueError("canonical category targets must partition required positions")
    return targets


def materialize_pretrain_stage(
    config: dict[str, Any],
    *,
    root: Path,
    stage_dir: Path,
    target_vocab: dict[str, int],
    target_offset: int,
    tokenize_and_encode: Callable[[str, str], tuple[list[str], list[int], dict[str, Any]]],
    eval_body_patterns: set[str],
) -> tuple[np.memmap, np.memmap, np.memmap, dict[str, Any]]:
    contract = config["data_model_scaling_contract"]
    targets = category_targets(contract)
    max_seq = int(config["tokenization"]["max_sequence_tokens"])
    index_path = stage_dir / "canonical_pretrain_index_v1.sqlite3"
    index_summary = build_document_index(config, root=root, index_path=index_path)
    outputs = pretrain_array_paths(stage_dir)
    temporary = {key: path.with_suffix(path.suffix + f".{os.getpid()}.tmp") for key, path in outputs.items()}
    consumed: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    category_row_ranges: dict[str, dict[str, int]] = {}
    excluded: Counter[str] = Counter()
    tokenizer_profiles: Counter[str] = Counter()
    tokenizer_category_profiles: Counter[str] = Counter()
    roundtrip_modes: Counter[str] = Counter()
    selected_digests: list[str] = []
    handles: dict[str, Any] = {}
    row_count = 0
    try:
        with temporary["inputs"].open("wb") as input_file, temporary["labels"].open("wb") as label_file, temporary["mask"].open("wb") as mask_file:
            with sqlite3.connect(index_path) as connection:
                for category in CATEGORY_ORDER:
                    category_start = row_count
                    cursor = connection.execute(
                        "SELECT digest, path, byte_offset, byte_length FROM documents WHERE category = ? ORDER BY digest",
                        (category,),
                    )
                    for digest, path_value, byte_offset, byte_length in cursor:
                        if consumed[category] >= targets[category]:
                            break
                        handle = handles.get(path_value)
                        if handle is None:
                            handle = Path(path_value).open("rb")
                            handles[path_value] = handle
                        handle.seek(int(byte_offset))
                        raw = handle.read(int(byte_length))
                        row = json.loads(raw)
                        text = str(row.get("text") if category in CATEGORY_ORDER[2:] else row.get("causal_text") or "")
                        logical_tokens, encoded, encoding_receipt = tokenize_and_encode(text, category)
                        if int(encoding_receipt.get("unknown_token_count") or 0):
                            excluded["tokenizer_unrepresentable"] += 1
                            continue
                        roundtrip = (
                            encoding_receipt.get("roundtrip")
                            if isinstance(encoding_receipt.get("roundtrip"), dict)
                            else {}
                        )
                        if roundtrip and roundtrip.get("state") != "GREEN":
                            excluded["tokenizer_roundtrip_failure"] += 1
                            continue
                        tokenizer_profiles[str(encoding_receipt.get("profile") or "unspecified")] += 1
                        tokenizer_category_profiles[
                            f"{category}:{encoding_receipt.get('profile') or 'unspecified'}"
                        ] += 1
                        roundtrip_modes[str(roundtrip.get("mode") or "not_reported")] += 1
                        normalized = " ".join(logical_tokens)
                        if any(pattern and pattern in normalized for pattern in eval_body_patterns):
                            excluded["eval_body_overlap"] += 1
                            continue
                        ids = [target_offset + int(value) for value in encoded]
                        document_used = False
                        for start in range(0, max(0, len(ids) - 1), max_seq):
                            remaining = targets[category] - consumed[category]
                            if remaining <= 0:
                                break
                            width = min(max_seq, len(ids) - start - 1, remaining)
                            if width <= 0:
                                break
                            inputs = np.zeros((max_seq,), dtype=np.int32)
                            labels = np.zeros((max_seq,), dtype=np.int32)
                            mask = np.zeros((max_seq,), dtype=np.uint8)
                            inputs[:width] = ids[start : start + width]
                            labels[:width] = ids[start + 1 : start + width + 1]
                            mask[:width] = 1
                            inputs.tofile(input_file)
                            labels.tofile(label_file)
                            mask.tofile(mask_file)
                            consumed[category] += width
                            window_counts[category] += 1
                            row_count += 1
                            document_used = True
                        if document_used:
                            selected_digests.append(str(digest))
                    category_row_ranges[category] = {
                        "start": category_start,
                        "stop": row_count,
                        "row_count": row_count - category_start,
                    }
        missing = {
            category: targets[category] - consumed[category]
            for category in CATEGORY_ORDER
            if consumed[category] != targets[category]
        }
        if missing:
            raise ValueError(f"canonical stage category targets not met: {missing}")
        for key in outputs:
            temporary[key].replace(outputs[key])
    finally:
        for handle in handles.values():
            handle.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
    shape = (row_count, max_seq)
    arrays = load_pretrain_memmaps(outputs, shape)
    report = {
        "policy": "project_theseus_canonical_balanced_pretrain_stage_v1",
        "category_targets": dict(targets),
        "category_positions": dict(consumed),
        "category_window_counts": dict(window_counts),
        "category_row_ranges": category_row_ranges,
        "arm_views": arm_views(config, category_row_ranges, consumed),
        "target_positions": sum(targets.values()),
        "materialized_positions": int(arrays[2].sum()),
        "window_count": row_count,
        "max_sequence_tokens": max_seq,
        "non_overlapping_windows": True,
        "index": index_summary,
        "selected_document_count": len(selected_digests),
        "selected_document_digest": sha256_text("\n".join(selected_digests)),
        "excluded_counts": dict(excluded),
        "tokenizer_audit": {
            "policy": "project_theseus_moecot_language_tokenizer_stage_audit_v1",
            "profiles_by_selected_document": dict(tokenizer_profiles),
            "category_profiles_by_selected_document": dict(tokenizer_category_profiles),
            "roundtrip_modes_by_selected_document": dict(roundtrip_modes),
            "roundtrip_failure_count": int(excluded.get("tokenizer_roundtrip_failure") or 0),
            "rejected_unknown_token_document_count": int(
                excluded.get("tokenizer_unrepresentable") or 0
            ),
            "admitted_unknown_token_position_count": 0,
            "failure_behavior": "reject_document_before_training",
        },
        "array_artifacts": {
            key: {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for key, path in outputs.items()
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    if report["materialized_positions"] != int(contract["required_unique_positions"]):
        raise ValueError("canonical stage position count mismatch")
    return arrays[0], arrays[1], arrays[2], report


def arm_views(
    config: dict[str, Any],
    category_ranges: dict[str, dict[str, int]],
    category_positions: Counter[str],
) -> dict[str, Any]:
    contract = config.get("moecot_language_seed_contract") or {}
    views: dict[str, Any] = {}
    for arm in contract.get("arms") or []:
        arm_id = str(arm.get("id") or "")
        categories = [str(value) for value in arm.get("categories") or []]
        if not arm_id or not categories or any(category not in category_ranges for category in categories):
            raise ValueError("invalid MoECOT arm category mapping")
        ranges = [category_ranges[category] for category in categories]
        views[arm_id] = {
            "categories": categories,
            "row_ranges": [{"start": row["start"], "stop": row["stop"]} for row in ranges],
            "row_count": sum(row["row_count"] for row in ranges),
            "target_positions": sum(int(category_positions[category]) for category in categories),
            "independent_weights_required": True,
        }
    if set(views) != {"english", "python", "javascript_typescript", "html_css", "rust"}:
        raise ValueError("canonical MoECOT language arm set is incomplete")
    return {
        "policy": "project_theseus_moecot_canonical_arm_views_v1",
        "arms": views,
        "mixed_dense_control": {
            "row_ranges": [{"start": 0, "stop": sum(row["row_count"] for row in category_ranges.values())}],
            "target_positions": sum(int(value) for value in category_positions.values()),
            "role": "matched_falsification_only",
        },
        "hidden_generalist_fallback": "forbidden",
    }


def pretrain_array_paths(stage_dir: Path) -> dict[str, Path]:
    return {
        "inputs": stage_dir / "canonical_pretrain_inputs_v1.i32",
        "labels": stage_dir / "canonical_pretrain_labels_v1.i32",
        "mask": stage_dir / "canonical_pretrain_mask_v1.u8",
    }


def load_pretrain_memmaps(
    outputs: dict[str, Path], shape: tuple[int, int], *, expected: dict[str, Any] | None = None,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    if expected:
        for key, path in outputs.items():
            row = expected.get(key) if isinstance(expected.get(key), dict) else {}
            if not path.is_file() or file_sha256(path) != str(row.get("sha256") or ""):
                raise ValueError(f"canonical pretrain array identity mismatch: {key}")
    return (
        np.memmap(outputs["inputs"], dtype=np.int32, mode="r", shape=shape),
        np.memmap(outputs["labels"], dtype=np.int32, mode="r", shape=shape),
        np.memmap(outputs["mask"], dtype=np.uint8, mode="r", shape=shape),
    )


def build_document_index(config: dict[str, Any], *, root: Path, index_path: Path) -> dict[str, Any]:
    corpus = config["canonical_corpus"]
    language_scope = validate_language_scope(corpus, root=root)
    quality_policy = validate_code_quality_policy(corpus, root=root)
    temporary = index_path.with_suffix(index_path.suffix + f".{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    counts: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    identities: list[dict[str, str]] = []
    code_deduper = ConversationDeduper(max_hamming_distance=int(corpus.get("near_duplicate_hamming_distance") or 3))
    english_deduper = ConversationDeduper(max_hamming_distance=int(corpus.get("near_duplicate_hamming_distance") or 3))
    with sqlite3.connect(temporary) as connection:
        connection.execute(
            "CREATE TABLE documents (category TEXT NOT NULL, digest TEXT NOT NULL, path TEXT NOT NULL, byte_offset INTEGER NOT NULL, byte_length INTEGER NOT NULL, PRIMARY KEY(category, digest))"
        )
        for manifest_value in corpus.get("code_shard_manifests") or []:
            manifest_path = resolve(root, manifest_value)
            manifest = read_json(manifest_path)
            shard_path = manifest_path.parent / str(manifest.get("sample_jsonl") or "")
            require_identity(shard_path, str(manifest.get("sample_jsonl_sha256") or ""))
            identities.extend(identity_rows(manifest_path, shard_path))
            for offset, length, row in iter_jsonl_ranges(shard_path):
                category = code_category(str(row.get("language") or ""), str(row.get("path") or ""))
                if not category:
                    continue
                text = str(row.get("text") or "")
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != str(row.get("text_sha256") or ""):
                    raise ValueError(f"code row content identity mismatch: {shard_path}:{offset}")
                quality_reasons = code_quality_rejection_reasons(
                    corpus, path=str(row.get("path") or ""), text=text, category=category
                )
                if quality_reasons:
                    for reason in quality_reasons:
                        exclusions[f"code_quality:{reason}"] += 1
                    continue
                add_index_row(
                    connection, code_deduper, category, text, shard_path, offset, length, counts, exclusions
                )
        conversation_root = resolve(root, corpus["conversation_root"])
        conversation_manifest = resolve(root, corpus["conversation_manifest"])
        manifest = read_json(conversation_manifest)
        identities.append(identity_row(conversation_manifest))
        for shard in manifest.get("shards") or []:
            shard_path = conversation_root / str(shard.get("train_path") or "")
            require_identity(shard_path, str(shard.get("train_sha256") or ""))
            identities.append(identity_row(shard_path))
            for offset, length, row in iter_jsonl_ranges(shard_path):
                text = str(row.get("causal_text") or "")
                add_index_row(
                    connection, english_deduper, "english_conversation_instruction", text,
                    shard_path, offset, length, counts, exclusions,
                )
        broad_root = resolve(root, corpus["broad_text_root"])
        broad_manifest = resolve(root, corpus["broad_text_manifest"])
        manifest = read_json(broad_manifest)
        identities.append(identity_row(broad_manifest))
        for shard in manifest.get("shards") or []:
            shard_path = broad_root / str(shard.get("train_path") or "")
            require_identity(shard_path, str(shard.get("train_sha256") or ""))
            identities.append(identity_row(shard_path))
            for offset, length, row in iter_jsonl_ranges(shard_path):
                text = str(row.get("causal_text") or "")
                add_index_row(
                    connection, english_deduper, "english_broad", text,
                    shard_path, offset, length, counts, exclusions,
                )
        connection.execute("CREATE INDEX documents_category_digest ON documents(category, digest)")
        connection.commit()
    temporary.replace(index_path)
    return {
        "path": str(index_path),
        "sha256": file_sha256(index_path),
        "document_counts": dict(counts),
        "excluded_counts": dict(exclusions),
        "source_identity_count": len(identities),
        "source_identity_digest": sha256_text(json.dumps(identities, sort_keys=True)),
        "language_scope": language_scope,
        "code_quality_policy": quality_policy,
    }


def validate_language_scope(corpus: dict[str, Any], *, root: Path) -> dict[str, Any]:
    scope = corpus.get("natural_language_scope") or {}
    allowed = [str(value).lower() for value in scope.get("allowed_languages") or []]
    if allowed != ["en"] or scope.get("non_allowed_action") != "quarantine":
        raise ValueError("canonical natural-language scope must be English-only and fail to quarantine")
    programming = list(corpus.get("programming_language_scope") or [])
    if programming != list(CATEGORY_ORDER[2:]):
        raise ValueError("canonical programming-language scope does not match category contract")
    policy_path = resolve(root, str(scope.get("intake_policy") or ""))
    policy = read_json(policy_path)
    enabled_conversation = [row for row in policy.get("sources") or [] if row.get("enabled") is True]
    enabled_documents = [row for row in policy.get("broad_text_sources") or [] if row.get("enabled") is True]
    invalid = []
    for row in enabled_conversation:
        if [str(value).lower() for value in row.get("required_langs") or []] != ["en"]:
            invalid.append(str(row.get("id") or "unknown"))
    for row in enabled_documents:
        if str(row.get("required_language") or "").lower() != "en":
            invalid.append(str(row.get("id") or "unknown"))
    if invalid:
        raise ValueError("enabled non-English natural-language intake source: " + ", ".join(invalid))
    return {
        "natural_languages": ["en"],
        "programming_languages": programming,
        "non_allowed_action": "quarantine",
        "intake_policy_path": str(policy_path),
        "intake_policy_sha256": file_sha256(policy_path),
        "enabled_conversation_source_count": len(enabled_conversation),
        "enabled_document_source_count": len(enabled_documents),
    }


def validate_code_quality_policy(corpus: dict[str, Any], *, root: Path) -> dict[str, Any]:
    policy = corpus.get("code_quality_policy") or {}
    if policy.get("policy") != "project_theseus_curated_code_quality_v1":
        raise ValueError("canonical code quality policy missing")
    if int(policy.get("minimum_logical_tokens") or 0) < 1:
        raise ValueError("canonical code quality token floor is invalid")
    if not 0.0 < float(policy.get("minimum_unique_token_ratio") or 0.0) <= 1.0:
        raise ValueError("canonical code quality diversity floor is invalid")
    repo_config = resolve(root, str(policy.get("curated_repo_config") or ""))
    repo_payload = read_json(repo_config)
    if not list(repo_payload.get("repos") or []):
        raise ValueError("canonical curated repository policy is empty")
    return {
        "policy": policy["policy"],
        "curated_repo_config": str(repo_config),
        "curated_repo_config_sha256": file_sha256(repo_config),
        "curated_repo_count": len(repo_payload["repos"]),
        "minimum_logical_tokens": int(policy["minimum_logical_tokens"]),
        "maximum_line_characters": int(policy["maximum_line_characters"]),
        "maximum_mean_nonempty_line_characters": float(
            policy["maximum_mean_nonempty_line_characters"]
        ),
        "minimum_unique_token_ratio": float(policy["minimum_unique_token_ratio"]),
    }


def code_quality_rejection_reasons(
    corpus: dict[str, Any], *, path: str, text: str, category: str
) -> list[str]:
    policy = corpus.get("code_quality_policy") or {}
    normalized_path = path.replace("\\", "/").lower()
    parts = {part for part in normalized_path.split("/") if part}
    reasons: list[str] = []
    if parts.intersection(str(value).lower() for value in policy.get("excluded_path_parts") or []):
        reasons.append("excluded_path")
    name = Path(normalized_path).name
    if any(str(marker).lower() in name for marker in policy.get("excluded_name_markers") or []):
        reasons.append("minified_or_bundle_name")
    prefix = text[:4096].lower()
    if any(str(marker).lower() in prefix for marker in policy.get("generated_text_markers") or []):
        reasons.append("generated_marker")
    if "\x00" in text or text.count("\ufffd") > max(1, len(text) // 10_000):
        reasons.append("binary_or_decode_damage")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        reasons.append("empty")
    else:
        if max(len(line) for line in lines) > int(policy.get("maximum_line_characters") or 0):
            reasons.append("extreme_line_length")
        if sum(len(line) for line in lines) / len(lines) > float(
            policy.get("maximum_mean_nonempty_line_characters") or 0.0
        ):
            reasons.append("minified_mean_line_length")
    logical = text.split()
    if len(logical) < int(policy.get("minimum_logical_tokens") or 0):
        reasons.append("too_few_logical_tokens")
    elif len(set(logical)) / len(logical) < float(policy.get("minimum_unique_token_ratio") or 0.0):
        reasons.append("low_token_diversity")
    if category == "python":
        try:
            compile(text, path or "<canonical-python>", "exec")
        except (SyntaxError, ValueError, TypeError):
            reasons.append("python_syntax_invalid")
    return list(dict.fromkeys(reasons))


def add_index_row(
    connection: sqlite3.Connection, deduper: ConversationDeduper, category: str, text: str,
    path: Path, offset: int, length: int, counts: Counter[str], exclusions: Counter[str],
) -> None:
    digest = sha256_text(" ".join(text.lower().split()))
    duplicate = deduper.classify(text, digest)
    if duplicate:
        exclusions[duplicate] += 1
        return
    deduper.add(text, digest)
    connection.execute(
        "INSERT INTO documents(category, digest, path, byte_offset, byte_length) VALUES (?, ?, ?, ?, ?)",
        (category, digest, str(path), offset, length),
    )
    counts[category] += 1


def iter_jsonl_ranges(path: Path) -> Iterator[tuple[int, int, dict[str, Any]]]:
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                return
            row = json.loads(raw)
            if isinstance(row, dict):
                yield offset, len(raw), row


def code_category(language: str, path: str) -> str:
    raw = f"{language} {Path(path).suffix}".lower()
    if "python" in raw or Path(path).suffix == ".py":
        return "python"
    if "typescript" in raw or "javascript" in raw or Path(path).suffix in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
        return "javascript_typescript"
    if "html" in raw or "css" in raw or Path(path).suffix in {".html", ".htm", ".css", ".scss", ".sass", ".less"}:
        return "html_css"
    if "rust" in raw or Path(path).suffix == ".rs":
        return "rust"
    return ""


def require_identity(path: Path, expected: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"canonical source identity mismatch: {path}")


def identity_rows(*paths: Path) -> list[dict[str, str]]:
    return [identity_row(path) for path in paths]


def identity_row(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
