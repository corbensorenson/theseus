#!/usr/bin/env python3
"""Build a bounded exact-text vocabulary shared by MoECOT arm ABIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from moecot_language_tokenizer import exact_text_tokens
from neural_seed_open_vocab import populate_open_vocab
from kernel_english_protocol import TRAINING_TASK_TAGS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moecot_language_arm_training.json"
SPECIAL = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
KERC_SOURCE_CONTROL_TOKENS = tuple(TRAINING_TASK_TAGS.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    cfg = validate_config(config)
    report = build(config, config_path=config_path) if args.execute else inspect(config, config_path)
    write_json(resolve(args.out or cfg["report"]), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PLANNED"} else 2


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("vocabulary") if isinstance(config.get("vocabulary"), dict) else {}
    if cfg.get("policy") != "project_theseus_moecot_exact_language_vocabulary_v1":
        raise ValueError("unexpected MoECOT vocabulary policy")
    if int(cfg.get("source_max_vocab") or 0) != 4096 or int(cfg.get("target_max_vocab") or 0) != 4096:
        raise ValueError("vocabulary size must preserve frozen model accounting")
    if tuple(cfg.get("required_source_control_tokens") or ()) != KERC_SOURCE_CONTROL_TOKENS:
        raise ValueError("trusted KERC source-control token contract mismatch")
    return cfg


def inspect(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = validate_config(config)
    path = resolve(cfg["output"])
    if not path.is_file():
        return report_base(config_path, cfg, "PLANNED", ["exact_language_vocabulary_missing"])
    payload = read_json(path)
    gaps = validate_payload(payload, cfg)
    return {
        **report_base(config_path, cfg, "RED" if gaps else "GREEN", gaps),
        "mode": "inspection",
        "vocabulary": artifact(path),
        "source_vocab_size": len(payload.get("source_vocab") or {}),
        "target_vocab_size": len(payload.get("target_vocab") or {}),
        "audit": payload.get("audit") or {},
    }


def build(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    cfg = validate_config(config)
    source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    source_docs = 0
    target_docs = 0
    input_artifacts: list[dict[str, Any]] = []

    code_path = resolve(cfg["code_samples"])
    input_artifacts.append(artifact(code_path))
    for row in read_jsonl(code_path):
        text = str(row.get("text") or "")
        if text:
            target_counts.update(exact_text_tokens(text))
            target_docs += 1

    conversation_root = resolve(cfg["conversation_root"])
    for path in sorted(conversation_root.glob("*.jsonl")):
        input_artifacts.append(artifact(path))
        for row in read_jsonl(path):
            for message in row.get("prompt_messages") or []:
                content = str(message.get("content") or "")
                if content:
                    source_counts.update(exact_text_tokens(content))
                    source_docs += 1
            target = row.get("target_message") if isinstance(row.get("target_message"), dict) else {}
            content = str(target.get("content") or "")
            if content:
                target_counts.update(exact_text_tokens(content))
                target_docs += 1

    broad_root = resolve(cfg["broad_english_root"])
    for path in sorted(broad_root.glob("*.jsonl")):
        input_artifacts.append(artifact(path))
        for row in read_jsonl(path):
            content = str(row.get("causal_text") or "")
            if content:
                target_counts.update(exact_text_tokens(content))
                target_docs += 1

    # Code excerpts occur on the prompt side of edit tasks. Add a bounded code
    # contribution without letting the much larger code corpus displace ordinary
    # English instruction tokens or creating a dependency on derived SFT rows.
    divisor = max(1, int(cfg.get("code_to_source_count_divisor") or 50))
    source_basis = Counter(source_counts)
    source_basis.update(
        {token: max(1, count // divisor) for token, count in target_counts.items()}
    )

    source_vocab = dict(SPECIAL)
    target_vocab = dict(SPECIAL)
    reserve_required_tokens(source_vocab, KERC_SOURCE_CONTROL_TOKENS)
    populate_open_vocab(
        source_vocab,
        source_basis,
        max_vocab=int(cfg["source_max_vocab"]),
        stream="source",
        piece_budget=int(cfg["byte_piece_budget"]),
    )
    populate_open_vocab(
        target_vocab,
        target_counts,
        max_vocab=int(cfg["target_max_vocab"]),
        stream="target",
        piece_budget=int(cfg["byte_piece_budget"]),
    )
    payload = {
        "policy": cfg["policy"],
        "created_utc": now(),
        "contract_sha256": contract_sha256(cfg),
        "source_vocab": source_vocab,
        "target_vocab": target_vocab,
        "audit": {
            "source_document_count": source_docs,
            "target_document_count": target_docs,
            "source_logical_token_count": sum(source_basis.values()),
            "target_logical_token_count": sum(target_counts.values()),
            "source_unique_logical_token_count": len(source_basis),
            "target_unique_logical_token_count": len(target_counts),
            "source_vocab_size": len(source_vocab),
            "target_vocab_size": len(target_vocab),
            "source_common_token_coverage": coverage(source_basis, source_vocab),
            "target_common_token_coverage": coverage(target_counts, target_vocab),
            "source_basis": "conversation_prompts_plus_open_code_counts_divided_by_50",
            "required_source_control_tokens": list(KERC_SOURCE_CONTROL_TOKENS),
            "trusted_control_tokens_observed_in_raw_text": False,
            "derived_supervision_rows_consumed": 0,
            "input_artifacts": input_artifacts,
            "public_benchmark_payload_count": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
    }
    path = resolve(cfg["output"])
    write_json_atomic(path, payload)
    gaps = validate_payload(payload, cfg)
    return {
        **report_base(config_path, cfg, "RED" if gaps else "GREEN", gaps),
        "mode": "materialized",
        "vocabulary": artifact(path),
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        "audit": payload["audit"],
    }


def coverage(counts: Counter[str], vocab: dict[str, int]) -> dict[str, Any]:
    total = sum(counts.values())
    atomic = sum(count for token, count in counts.items() if token in vocab)
    return {
        "logical_token_count": total,
        "atomic_token_count": atomic,
        "atomic_token_ratio": round(atomic / max(1, total), 8),
        "byte_fallback_logical_token_count": total - atomic,
    }


def validate_payload(payload: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    if payload.get("policy") != cfg["policy"]:
        gaps.append("vocabulary_policy_mismatch")
    if payload.get("contract_sha256") != contract_sha256(cfg):
        gaps.append("vocabulary_contract_identity_mismatch")
    for stream in ("source", "target"):
        vocab = payload.get(f"{stream}_vocab") if isinstance(payload.get(f"{stream}_vocab"), dict) else {}
        if len(vocab) != int(cfg[f"{stream}_max_vocab"]):
            gaps.append(f"{stream}_vocabulary_size_mismatch")
        if len(set(vocab.values())) != len(vocab):
            gaps.append(f"{stream}_vocabulary_id_collision")
        for token in (f"<{stream}_token_bytes>", f"</{stream}_token_bytes>"):
            if token not in vocab:
                gaps.append(f"{stream}_byte_boundary_missing")
        if not all(f"<byte:{value:02x}>" in vocab for value in range(256)):
            gaps.append(f"{stream}_byte_inventory_incomplete")
        required = cfg.get(f"required_{stream}_control_tokens") or []
        for token in required:
            if token not in vocab:
                gaps.append(f"{stream}_required_control_token_missing:{token}")
    return gaps


def reserve_required_tokens(vocab: dict[str, int], tokens: Iterable[str]) -> None:
    for token in tokens:
        value = str(token)
        if not value or value in vocab:
            raise ValueError(f"required vocabulary token invalid or duplicated: {value}")
        vocab[value] = len(vocab)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": relative(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def report_base(config_path: Path, cfg: dict[str, Any], state: str, gaps: list[str]) -> dict[str, Any]:
    return {
        "policy": cfg["policy"],
        "created_utc": now(),
        "trigger_state": state,
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "contract_sha256": contract_sha256(cfg),
        "hard_gaps": gaps,
        "score_semantics": "tokenization efficiency and reversibility only; not capability",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def contract_sha256(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
