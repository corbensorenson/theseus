#!/usr/bin/env python3
"""Governed narrow English+code corpus, tokenizer, and pretraining spine.

This is the missing stage before task-pair adaptation: Theseus needs a
from-scratch language/code foundation before strict code-body supervision can
reasonably transfer to held-out task families. It admits only manifest-governed
license-clean English/code text, trains a from-scratch BPE tokenizer, pretrains
matched SymLiquid-style and transformer language-model arms, and writes
checkpoint/tokenizer artifacts that the strict comparator can consume later.

It never loads open/base/pretrained model weights, runs public calibration,
trains on public benchmark payloads, serves external tokens, or credits
tool/template output as learned generation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "narrow_corpus_pretraining_spine.json"
DEFAULT_OUT = ROOT / "reports" / "narrow_corpus_pretraining_spine.json"
DEFAULT_MANIFEST = ROOT / "data" / "training_sources" / "narrow_corpus_manifest.json"
DEFAULT_TOKENIZER = ROOT / "checkpoints" / "narrow_pretraining" / "tokenizer.json"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "narrow_pretraining"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PUBLIC_BENCHMARK_TOKENS = {
    "mbpp",
    "evalplus",
    "human_eval",
    "humaneval",
    "bigcodebench",
    "livecodebench",
    "swe_bench",
    "swe-bench",
    "terminal_bench",
    "webarena",
    "mmlu",
    "gpqa",
    "gsm8k",
    "ruler",
    "babilong",
    "longbench",
    "longmemeval",
    "needlebench",
    "infinitebench",
    "blimp",
}
PUBLIC_PAYLOAD_HINTS = {
    "canonical_solution",
    "hidden_tests",
    "public_tests",
    "test_list",
    "assert ",
    "assert(",
    "expected_output",
    "task_id",
    "benchmark",
}
ALLOWED_LICENSES = {
    "project-internal",
    "corben-owned",
    "psf-2.0",
    "python-software-foundation-license-2.0",
    "mit",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "apache-2.0",
    "public-domain",
    "cc0",
    "0bsd",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--manifest-out", default=rel(DEFAULT_MANIFEST))
    parser.add_argument("--tokenizer-out", default=rel(DEFAULT_TOKENIZER))
    parser.add_argument("--checkpoint-dir", default=rel(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_config(resolve(args.config))
    report = run_spine(
        config,
        config_path=args.config,
        out=resolve(args.out),
        manifest_out=resolve(args.manifest_out),
        tokenizer_out=resolve(args.tokenizer_out),
        checkpoint_dir=resolve(args.checkpoint_dir),
        execute=bool(args.execute),
        started=started,
    )
    write_json(resolve(args.out), report)
    print(json.dumps(strip_tensor_free(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW", "PLANNED"} else 2


def run_spine(
    config: dict[str, Any],
    *,
    config_path: str,
    out: Path,
    manifest_out: Path,
    tokenizer_out: Path,
    checkpoint_dir: Path,
    execute: bool,
    started: float,
) -> dict[str, Any]:
    corpus_cfg = dict_or_empty(config.get("corpus"))
    tokenizer_cfg = dict_or_empty(config.get("tokenizer"))
    pretrain_cfg = dict_or_empty(config.get("pretraining"))
    admissions = admit_corpus(corpus_cfg, config=config)
    admitted = [row for row in admissions if row["admitted"]]
    manifest = corpus_manifest(config, admissions)
    write_json(manifest_out, manifest)
    if not execute:
        return {
            "policy": "project_theseus_narrow_corpus_pretraining_spine_v1",
            "created_utc": now(),
            "config": config_path,
            "execute": False,
            "trigger_state": "PLANNED",
            "summary": {
                "admitted_document_count": len(admitted),
                "admitted_char_count": sum(int(row["char_count"]) for row in admitted),
                "admitted_rough_token_count": sum(int(row["rough_token_count"]) for row in admitted),
                "content_token_breakdown": content_token_breakdown(admitted),
                "license_breakdown": value_breakdown(admitted, "license"),
                "public_benchmark_payload_admitted": any(row["public_benchmark_payload_detected"] for row in admitted),
                "eval_overlap_admitted": any(row.get("eval_overlap_detected") for row in admitted),
                "manifest": rel(manifest_out),
                "tokenizer_out": rel(tokenizer_out),
                "checkpoint_dir": rel(checkpoint_dir),
            },
            "corpus_manifest": rel(manifest_out),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        }

    docs = [str(row["text"]) for row in admitted]
    tokenizer = train_bpe_tokenizer(
        docs,
        target_vocab_size=int(tokenizer_cfg.get("target_vocab_size") or 8192),
        min_pair_frequency=int(tokenizer_cfg.get("min_pair_frequency") or 2),
        max_merges=int(tokenizer_cfg.get("max_merges") or 8192),
        max_train_chars=int(tokenizer_cfg.get("max_train_chars") or 1_200_000),
    )
    tokenizer_payload = tokenizer.to_json()
    tokenizer_payload["corpus_manifest_sha256"] = sha256_json(manifest)
    tokenizer_payload["config"] = {
        "target_vocab_size": int(tokenizer_cfg.get("target_vocab_size") or 8192),
        "min_pair_frequency": int(tokenizer_cfg.get("min_pair_frequency") or 2),
        "max_merges": int(tokenizer_cfg.get("max_merges") or 8192),
    }
    write_json(tokenizer_out, tokenizer_payload)

    encoded_train, encoded_eval = encode_corpus_split(
        docs,
        tokenizer,
        max_tokens=int(pretrain_cfg.get("max_total_tokens") or 200_000),
        eval_fraction=float(pretrain_cfg.get("eval_fraction") or 0.10),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pretraining_reports: list[dict[str, Any]] = []
    if len(encoded_train) >= 128 and len(tokenizer.vocab) >= int(tokenizer_cfg.get("min_vocab_size") or 256):
        import torch  # Imported lazily so corpus/tokenizer audit stays lightweight.

        device = choose_torch_device(torch, str(pretrain_cfg.get("device") or "auto"))
        default_arms = [
            str(arm)
            for arm in (pretrain_cfg.get("arms") or ["transformer_control", "symliquid_style"])
            if str(arm) in {"transformer_control", "symliquid_style"}
        ] or ["transformer_control", "symliquid_style"]
        for budget in list(pretrain_cfg.get("budgets") or default_budgets()):
            budget = dict_or_empty(budget)
            budget_arms = [
                str(arm)
                for arm in (budget.get("arms") or default_arms)
                if str(arm) in {"transformer_control", "symliquid_style"}
            ] or default_arms
            for arm_id in budget_arms:
                pretraining_reports.append(
                    pretrain_arm(
                        arm_id,
                        budget,
                        encoded_train,
                        encoded_eval,
                        tokenizer,
                        checkpoint_dir=checkpoint_dir,
                        manifest_path=manifest_out,
                        tokenizer_path=tokenizer_out,
                        torch=torch,
                        device=device,
                        seed=int(pretrain_cfg.get("seed") or 23),
                    )
                )

    gates = build_gates(admissions, tokenizer, encoded_train, pretraining_reports)
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warn_failed = [row for row in gates if row["severity"] != "hard" and not row["passed"]]
    trigger_state = "GREEN" if not hard_failed else "RED"
    if trigger_state == "GREEN" and warn_failed:
        trigger_state = "YELLOW"
    report = {
        "policy": "project_theseus_narrow_corpus_pretraining_spine_v1",
        "created_utc": now(),
        "config": config_path,
        "execute": True,
        "trigger_state": trigger_state,
        "summary": {
            "admitted_document_count": len(admitted),
            "rejected_document_count": len([row for row in admissions if not row["admitted"]]),
            "admitted_char_count": sum(int(row["char_count"]) for row in admitted),
            "admitted_rough_token_count": sum(int(row["rough_token_count"]) for row in admitted),
            "admitted_token_count": len(encoded_train) + len(encoded_eval),
            "train_token_count": len(encoded_train),
            "eval_token_count": len(encoded_eval),
            "content_token_breakdown": content_token_breakdown(admitted),
            "content_document_breakdown": value_breakdown(admitted, "content_type"),
            "license_breakdown": value_breakdown(admitted, "license"),
            "source_kind_breakdown": value_breakdown(admitted, "source_kind"),
            "algorithmic_python_token_fraction": algorithmic_python_token_fraction(admitted),
            "tokenizer_vocab_size": len(tokenizer.vocab),
            "tokenizer_merge_count": len(tokenizer.merges),
            "tokenizer_sha256": sha256_json(tokenizer_payload),
            "corpus_manifest": rel(manifest_out),
            "corpus_manifest_sha256": sha256_json(manifest),
            "tokenizer": rel(tokenizer_out),
            "checkpoint_dir": rel(checkpoint_dir),
            "pretraining_runs": len(pretraining_reports),
            "completed_pretraining_runs": sum(1 for row in pretraining_reports if row.get("ok")),
            "max_optimizer_token_positions_consumed": max(
                [int(row.get("token_positions_consumed") or 0) for row in pretraining_reports if row.get("ok")] or [0]
            ),
            "max_optimizer_windows_consumed": max(
                [int(row.get("windows_consumed") or 0) for row in pretraining_reports if row.get("ok")] or [0]
            ),
            "transformer_completed_pretraining_runs": sum(
                1 for row in pretraining_reports if row.get("ok") and str(row.get("arm_id")) == "transformer_control"
            ),
            "transformer_max_parameter_update_fraction": max(
                [
                    float(row.get("parameter_update_fraction") or 0.0)
                    for row in pretraining_reports
                    if row.get("ok") and str(row.get("arm_id")) == "transformer_control"
                ]
                or [0.0]
            ),
            "transformer_max_non_embedding_update_fraction": max(
                [
                    float(row.get("non_embedding_update_fraction") or 0.0)
                    for row in pretraining_reports
                    if row.get("ok") and str(row.get("arm_id")) == "transformer_control"
                ]
                or [0.0]
            ),
            "transformer_heldout_lm_improved_count": sum(
                1
                for row in pretraining_reports
                if row.get("ok")
                and str(row.get("arm_id")) == "transformer_control"
                and bool(row.get("heldout_lm_improved"))
            ),
            "public_benchmark_payload_admitted": any(row["public_benchmark_payload_detected"] for row in admitted),
            "eval_overlap_admitted": any(row.get("eval_overlap_detected") for row in admitted),
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "open_or_pretrained_model_weights_used": False,
        },
        "score_semantics": (
            "Self-supervised from-scratch pretraining canary over admitted narrow corpus only. "
            "This is not public calibration, not task-pair adaptation, and not promotion evidence."
        ),
        "hard_invariants": [
            "No open/base/pretrained weights are loaded.",
            "Public benchmark prompts/tests/solutions/traces/templates remain calibration-only and are not admitted.",
            "External inference calls are zero.",
            "Checkpoints are from random initialization on the admitted corpus.",
        ],
        "gates": gates,
        "corpus_admissions": summarize_admissions(admissions),
        "tokenizer_summary": tokenizer_summary(tokenizer),
        "pretraining": pretraining_reports,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
    write_json(out, report)
    return report


def admit_corpus(corpus_config: dict[str, Any], *, config: dict[str, Any]) -> list[dict[str, Any]]:
    source_specs = corpus_source_specs(corpus_config)
    global_suffixes = {str(item).lower() for item in corpus_config.get("suffixes", default_suffixes())}
    global_excluded_dirs = set(corpus_config.get("excluded_dirs", default_excluded_dirs()))
    max_files = int(corpus_config.get("max_files") or 512)
    max_chars_per_file = int(corpus_config.get("max_chars_per_file") or 40_000)
    max_total_chars = int(corpus_config.get("max_total_chars") or 1_500_000)
    firewall = build_text_firewall(config)
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    seen_sha256: set[str] = set()
    seen_normalized_sha256: set[str] = set()
    total_chars = 0
    for spec in source_specs:
        root = resolve(str(spec.get("root") or ""))
        suffixes = {str(item).lower() for item in spec.get("suffixes", global_suffixes)}
        excluded_dirs = global_excluded_dirs | set(spec.get("excluded_dirs", []))
        source_max_files = int(spec.get("max_files") or max_files)
        source_max_chars_per_file = int(spec.get("max_chars_per_file") or max_chars_per_file)
        source_max_total_chars = int(spec.get("max_total_chars") or max_total_chars)
        source_admitted = 0
        source_chars = 0
        candidates = [root] if root.is_file() else sorted(root.rglob("*")) if root.exists() else []
        for path in candidates:
            if len([row for row in rows if row["admitted"]]) >= max_files or total_chars >= max_total_chars:
                break
            if source_admitted >= source_max_files or source_chars >= source_max_total_chars:
                break
            if path in seen_paths or not path.is_file():
                continue
            seen_paths.add(path)
            rel_path = rel(path)
            if path_has_excluded_part(path, root, excluded_dirs):
                rows.append(reject_row(path, "excluded_directory", spec=spec))
                continue
            if path.suffix.lower() not in suffixes:
                continue
            license_id = normalize_license(str(spec.get("license") or corpus_config.get("project_internal_license") or ""))
            license_ok = license_allowed(license_id)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                rows.append(reject_row(path, f"read_error:{exc.__class__.__name__}", spec=spec))
                continue
            if not text.strip():
                rows.append(reject_row(path, "empty_text", spec=spec))
                continue
            text = text[:source_max_chars_per_file]
            public_refs = public_reference_tokens(rel_path, text)
            payload_detected = public_payload_detected(rel_path, text, public_refs)
            eval_overlap = eval_overlap_detected(text, firewall)
            digest = sha256_text(text)
            normalized_digest = sha256_text(normalize_for_dedupe(text))
            duplicate = digest in seen_sha256 or normalized_digest in seen_normalized_sha256
            admitted = license_ok and not payload_detected and not eval_overlap and not duplicate
            if admitted:
                reason = "admitted"
            elif not license_ok:
                reason = "unknown_or_disallowed_license"
            elif payload_detected:
                reason = "public_benchmark_payload_detected"
            elif eval_overlap:
                reason = "eval_or_public_calibration_overlap_detected"
            else:
                reason = "duplicate_sha256_or_normalized_sha256"
            token_count = rough_token_count(text)
            content_type = str(spec.get("content_type") or infer_content_type(path))
            algorithmic_python = bool(path.suffix == ".py" and looks_algorithmic_python(text))
            row = {
                "source_id": f"local:{digest[:16]}",
                "path": rel_path,
                "source_spec_id": str(spec.get("id") or stable_hash(str(root))[:12]),
                "source_kind": str(spec.get("source_kind") or "project_internal_text_or_code"),
                "content_type": content_type,
                "algorithmic_python": algorithmic_python,
                "license": license_id,
                "license_allowed": license_ok,
                "license_file": rel(resolve(str(spec.get("license_file")))) if spec.get("license_file") else "",
                "provenance": str(spec.get("provenance") or "local_project_workspace"),
                "admitted": admitted,
                "training_use": "self_supervised_pretraining" if admitted else "rejected",
                "reason": reason,
                "sha256": digest,
                "normalized_sha256": normalized_digest,
                "char_count": len(text),
                "rough_token_count": token_count,
                "public_reference_tokens": sorted(public_refs),
                "public_benchmark_payload_detected": payload_detected,
                "eval_overlap_detected": eval_overlap,
                "eval_overlap_hit_count": eval_overlap_hit_count(text, firewall),
                "dedupe_status": "duplicate" if duplicate else "unique",
                "text": text if admitted else "",
            }
            rows.append(row)
            if admitted:
                total_chars += len(text)
                source_chars += len(text)
                source_admitted += 1
                seen_sha256.add(digest)
                seen_normalized_sha256.add(normalized_digest)
    return rows


def reject_row(path: Path, reason: str, *, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = dict_or_empty(spec)
    return {
        "source_id": f"local_rejected:{stable_hash(rel(path))[:16]}",
        "path": rel(path),
        "source_spec_id": str(spec.get("id") or ""),
        "source_kind": str(spec.get("source_kind") or "project_internal_text_or_code"),
        "content_type": str(spec.get("content_type") or infer_content_type(path)),
        "algorithmic_python": False,
        "license": normalize_license(str(spec.get("license") or "project-internal")),
        "license_allowed": license_allowed(str(spec.get("license") or "project-internal")),
        "license_file": rel(resolve(str(spec.get("license_file")))) if spec.get("license_file") else "",
        "provenance": str(spec.get("provenance") or "local_project_workspace"),
        "admitted": False,
        "training_use": "rejected",
        "reason": reason,
        "sha256": "",
        "normalized_sha256": "",
        "char_count": 0,
        "rough_token_count": 0,
        "public_reference_tokens": [],
        "public_benchmark_payload_detected": False,
        "eval_overlap_detected": False,
        "eval_overlap_hit_count": 0,
        "dedupe_status": "not_applicable",
        "text": "",
    }


def corpus_source_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = config.get("sources")
    if isinstance(specs, list) and specs:
        return [dict_or_empty(row) for row in specs if dict_or_empty(row).get("root")]
    return [
        {
            "id": f"legacy_root_{idx}",
            "root": root,
            "source_kind": "project_internal_text_or_code",
            "content_type": "mixed_project_text_code",
            "license": str(config.get("project_internal_license") or "project-internal"),
            "provenance": "local_project_workspace",
        }
        for idx, root in enumerate(config.get("roots", default_roots()))
    ]


def path_has_excluded_part(path: Path, root: Path, excluded_dirs: set[str]) -> bool:
    if not excluded_dirs:
        return False
    try:
        parts = path.relative_to(root if root.is_dir() else root.parent).parts
    except ValueError:
        try:
            parts = path.relative_to(ROOT).parts
        except ValueError:
            parts = path.parts
    return any(part in excluded_dirs for part in parts)


def normalize_license(license_id: str) -> str:
    return re.sub(r"[^a-z0-9.+-]+", "-", license_id.strip().lower()).strip("-")


def license_allowed(license_id: str) -> bool:
    return normalize_license(license_id) in ALLOWED_LICENSES


def infer_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "code_python"
    if suffix == ".rs":
        return "code_rust"
    if suffix in {".sh", ".bash", ".zsh"}:
        return "code_shell"
    if suffix in {".md", ".txt", ".rst"}:
        return "english_text"
    if suffix in {".json", ".jsonl", ".toml", ".yaml", ".yml"}:
        return "structured_config"
    return "other_text"


def looks_algorithmic_python(text: str) -> bool:
    try:
        parsed = ast.parse(text)
    except SyntaxError:
        return False
    function_count = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(parsed))
    class_count = sum(isinstance(node, ast.ClassDef) for node in ast.walk(parsed))
    control_nodes = [ast.For, ast.While, ast.If, ast.Try, ast.With]
    match_node = getattr(ast, "Match", None)
    if match_node is not None:
        control_nodes.append(match_node)
    control_count = sum(isinstance(node, tuple(control_nodes)) for node in ast.walk(parsed))
    return function_count >= 1 and (control_count >= 2 or class_count >= 1)


def normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def build_text_firewall(config: dict[str, Any]) -> dict[str, Any]:
    firewall_cfg = dict_or_empty(config.get("firewall"))
    paths: list[str] = []
    if bool(firewall_cfg.get("include_strict_comparator_eval", True)):
        comparator_path = resolve(str(firewall_cfg.get("strict_comparator_config") or "configs/neural_seed_token_decoder_comparator.json"))
        comparator = read_json(comparator_path) if comparator_path.exists() else {}
        eval_path = get_path(comparator, ["data", "eval_jsonl"], "")
        if eval_path:
            paths.append(str(eval_path))
    for item in firewall_cfg.get("eval_jsonl", []):
        paths.append(str(item))
    strings: list[str] = []
    hashes: set[str] = set()
    for path_text in sorted(set(paths)):
        path = resolve(path_text)
        for row in read_jsonl(path):
            for key in [
                "prompt",
                "instruction",
                "entry_point",
                "solution",
                "solution_body",
                "canonical_solution",
                "tests",
                "hidden_tests",
                "expected",
                "answer",
            ]:
                value = row.get(key)
                if isinstance(value, str):
                    stripped = value.strip()
                    if len(stripped) >= 32:
                        hashes.add(sha256_text(stripped))
                    if len(stripped) >= 80:
                        strings.append(stripped[:4000])
    return {
        "policy": "project_theseus_corpus_firewall_v1",
        "source_paths": sorted(set(paths)),
        "exact_string_count": len(strings),
        "hash_count": len(hashes),
        "exact_strings": strings,
        "hashes": hashes,
    }


def eval_overlap_detected(text: str, firewall: dict[str, Any]) -> bool:
    return eval_overlap_hit_count(text, firewall) > 0


def eval_overlap_hit_count(text: str, firewall: dict[str, Any]) -> int:
    hits = 0
    haystack = text[:250000]
    for forbidden in firewall.get("exact_strings", []):
        if forbidden and forbidden in haystack:
            hits += 1
    stripped = text.strip()
    if sha256_text(stripped) in firewall.get("hashes", set()):
        hits += 1
    return hits


def corpus_manifest(config: dict[str, Any], admissions: list[dict[str, Any]]) -> dict[str, Any]:
    admitted = [row for row in admissions if row["admitted"]]
    public_payload = [row for row in admitted if row["public_benchmark_payload_detected"]]
    eval_overlap = [row for row in admitted if row.get("eval_overlap_detected")]
    rows = []
    for row in admissions:
        copied = {key: value for key, value in row.items() if key != "text"}
        rows.append(copied)
    return {
        "policy": "project_theseus_narrow_corpus_manifest_v1",
        "created_utc": now(),
        "scope": dict_or_empty(config.get("scope")) or {
            "include": ["English prose", "Python/code", "Project Theseus docs/source/configs"],
            "exclude": ["public benchmark payloads", "multilingual bulk text", "trivia-first corpora", "uncertain-license data"],
        },
        "summary": {
            "source_count": len(admissions),
            "admitted_document_count": len(admitted),
            "admitted_char_count": sum(int(row["char_count"]) for row in admitted),
            "admitted_rough_token_count": sum(int(row["rough_token_count"]) for row in admitted),
            "content_token_breakdown": content_token_breakdown(admitted),
            "license_breakdown": value_breakdown(admitted, "license"),
            "source_kind_breakdown": value_breakdown(admitted, "source_kind"),
            "algorithmic_python_token_fraction": algorithmic_python_token_fraction(admitted),
            "public_benchmark_payload_admitted_count": len(public_payload),
            "eval_overlap_admitted_count": len(eval_overlap),
            "duplicate_rejected_count": sum(1 for row in admissions if str(row.get("reason")) == "duplicate_sha256_or_normalized_sha256"),
            "public_training_rows": 0,
            "external_inference_calls": 0,
        },
        "hard_invariants": [
            "No public benchmark prompts, tests, hidden tests, solutions, traces, or answer templates are admitted.",
            "No open/base/pretrained model weights are used.",
            "The corpus is for self-supervised pretraining only; public benchmarks remain calibration-only.",
        ],
        "sources": rows,
    }


@dataclass
class BpeTokenizer:
    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    _piece_cache: dict[tuple[str, ...], list[str]] = field(default_factory=dict, repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "policy": "project_theseus_bpe_tokenizer_v1",
            "created_utc": now(),
            "special_tokens": SPECIAL_TOKENS,
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
            "vocab_size": len(self.vocab),
            "merge_count": len(self.merges),
            "from_scratch": True,
            "open_or_pretrained_tokenizer_used": False,
        }

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        ranks = {pair: idx for idx, pair in enumerate(self.merges)}
        for token in basic_tokens(text):
            key = tuple(token)
            pieces = self._piece_cache.get(key)
            if pieces is None:
                pieces = apply_merges(key, ranks)
                self._piece_cache[key] = pieces
            for piece in pieces:
                ids.append(self.vocab.get(piece, self.vocab["<unk>"]))
        return ids


def train_bpe_tokenizer(
    docs: list[str],
    *,
    target_vocab_size: int,
    min_pair_frequency: int,
    max_merges: int,
    max_train_chars: int,
) -> BpeTokenizer:
    text = "\n<doc>\n".join(docs)[:max_train_chars]
    word_counts: Counter[tuple[str, ...]] = Counter()
    for token in basic_tokens(text):
        if token:
            word_counts[tuple(token)] += 1
    vocab_symbols = set(SPECIAL_TOKENS)
    for word in word_counts:
        vocab_symbols.update(word)
    merges: list[tuple[str, str]] = []
    max_new = max(0, min(max_merges, target_vocab_size - len(vocab_symbols)))
    for _ in range(max_new):
        pairs: Counter[tuple[str, str]] = Counter()
        for word, count in word_counts.items():
            for idx in range(len(word) - 1):
                pairs[(word[idx], word[idx + 1])] += count
        if not pairs:
            break
        pair, freq = pairs.most_common(1)[0]
        if freq < min_pair_frequency:
            break
        merged = "".join(pair)
        new_counts: Counter[tuple[str, ...]] = Counter()
        for word, count in word_counts.items():
            new_counts[merge_pair_in_word(word, pair, merged)] += count
        word_counts = new_counts
        merges.append(pair)
        vocab_symbols.add(merged)
        if len(vocab_symbols) >= target_vocab_size:
            break
    piece_counts: Counter[str] = Counter()
    for word, count in word_counts.items():
        for piece in word:
            piece_counts[piece] += count
    ordered = list(SPECIAL_TOKENS)
    for piece, _ in piece_counts.most_common():
        if piece not in ordered:
            ordered.append(piece)
        if len(ordered) >= target_vocab_size:
            break
    return BpeTokenizer(vocab={piece: idx for idx, piece in enumerate(ordered)}, merges=merges)


def basic_tokens(text: str) -> list[list[str]]:
    out: list[list[str]] = []
    for raw in re.findall(r"\n|[ \t]+|[A-Za-z_][A-Za-z0-9_]*|\d+|[^\w\s]", text):
        if raw == "\n":
            out.append(["<nl>"])
        elif raw.isspace():
            out.append(["<sp>"] if "\t" not in raw else ["<tab>"])
        elif re.match(r"[A-Za-z_][A-Za-z0-9_]*$", raw):
            out.append(["_"] + list(raw))
        else:
            out.append(list(raw))
    return out


def merge_pair_in_word(word: tuple[str, ...], pair: tuple[str, str], merged: str) -> tuple[str, ...]:
    out: list[str] = []
    idx = 0
    while idx < len(word):
        if idx < len(word) - 1 and word[idx] == pair[0] and word[idx + 1] == pair[1]:
            out.append(merged)
            idx += 2
        else:
            out.append(word[idx])
            idx += 1
    return tuple(out)


def apply_merges(word: tuple[str, ...], ranks: dict[tuple[str, str], int]) -> list[str]:
    pieces = list(word)
    while len(pieces) > 1:
        best_index = -1
        best_rank = 10**12
        for idx in range(len(pieces) - 1):
            rank = ranks.get((pieces[idx], pieces[idx + 1]))
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_index = idx
        if best_index < 0:
            break
        pieces[best_index : best_index + 2] = [pieces[best_index] + pieces[best_index + 1]]
    return pieces


def encode_corpus_split(
    docs: list[str],
    tokenizer: BpeTokenizer,
    *,
    max_tokens: int,
    eval_fraction: float,
) -> tuple[list[int], list[int]]:
    eos = tokenizer.vocab["<eos>"]
    ids: list[int] = []
    for doc in docs:
        ids.extend(tokenizer.encode(doc))
        ids.append(eos)
        if len(ids) >= max_tokens:
            break
    ids = ids[:max_tokens]
    split = max(1, int(len(ids) * (1.0 - max(0.01, min(0.5, eval_fraction)))))
    return ids[:split], ids[split:] or ids[-min(len(ids), 128):]


def pretrain_arm(
    arm_id: str,
    budget: dict[str, Any],
    train_tokens: list[int],
    eval_tokens: list[int],
    tokenizer: BpeTokenizer,
    *,
    checkpoint_dir: Path,
    manifest_path: Path,
    tokenizer_path: Path,
    torch: Any,
    device: Any,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    vocab_size = len(tokenizer.vocab)
    seq_len = int(budget.get("seq_len") or 64)
    steps = int(budget.get("steps") or 64)
    batch_size = int(budget.get("batch_size") or 16)
    lr = float(budget.get("learning_rate") or 0.002)
    d_model = int(budget.get("d_model") or 64)
    if arm_id == "transformer_control":
        model = TransformerLm(
            vocab_size,
            d_model=d_model,
            nhead=int(budget.get("nhead") or 4),
            num_layers=int(budget.get("num_layers") or 1),
            dim_feedforward=int(budget.get("dim_feedforward") or d_model * 2),
            max_len=seq_len,
            torch=torch,
        )
    else:
        model = SymLiquidLm(
            vocab_size,
            hidden_dim=d_model,
            reservoir_dim=int(budget.get("reservoir_dim") or d_model * 2),
            hv_dim=int(budget.get("hv_dim") or d_model * 2),
            torch=torch,
        )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(budget.get("weight_decay") or 0.0001))
    losses: list[float] = []
    started = time.perf_counter()
    before = model_parameter_snapshot(model, torch=torch)
    eval_loss_before = evaluate_lm(model, eval_tokens, seq_len, batch_size, torch=torch, device=device)
    for _step in range(steps):
        x, y = sample_lm_batch(train_tokens, seq_len, batch_size, torch=torch, device=device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(round(float(loss.detach().cpu()), 6))
    eval_loss = evaluate_lm(model, eval_tokens, seq_len, batch_size, torch=torch, device=device)
    update_summary = model_parameter_update_summary(model, before, torch=torch)
    checkpoint_path = checkpoint_dir / f"{arm_id}_{budget.get('id', 'canary')}.pt"
    train_window_count = max(0, len(train_tokens) - seq_len)
    eval_window_count = max(0, len(eval_tokens) - seq_len)
    windows_consumed = int(steps) * int(batch_size)
    token_positions_consumed = windows_consumed * int(seq_len)
    train_loss_curve_max_points = max(0, int(budget.get("loss_curve_max_points") or 0))
    train_loss_curve = compact_loss_curve(losses, train_loss_curve_max_points)
    torch.save(
        {
            "policy": "project_theseus_from_scratch_lm_pretraining_checkpoint_v1",
            "created_utc": now(),
            "arm_id": arm_id,
            "budget": budget,
            "vocab_size": vocab_size,
            "embedding_key": "embedding.weight",
            "model_state_dict": model.state_dict(),
            "tokenizer": rel(tokenizer_path),
            "tokenizer_sha256": sha256_json(tokenizer.to_json()),
            "corpus_manifest": rel(manifest_path),
            "corpus_manifest_sha256": sha256_file(manifest_path),
            "open_or_pretrained_model_weights_used": False,
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "audit": {
                "trainable_parameter_count": update_summary["trainable_parameter_count"],
                "parameter_update_fraction": update_summary["parameter_update_fraction"],
                "non_embedding_update_fraction": update_summary["non_embedding_update_fraction"],
                "optimizer_step_count": steps,
                "windows_consumed": windows_consumed,
                "token_positions_consumed": token_positions_consumed,
                "open_or_pretrained_model_weights_used": False,
                "public_training_rows": 0,
                "external_inference_calls": 0,
            },
        },
        checkpoint_path,
    )
    return {
        "ok": True,
        "arm_id": arm_id,
        "budget_id": str(budget.get("id") or "canary"),
        "checkpoint": rel(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": count_params(model),
        "trainable_parameter_count": update_summary["trainable_parameter_count"],
        "updated_parameter_count": update_summary["updated_parameter_count"],
        "parameter_update_fraction": update_summary["parameter_update_fraction"],
        "non_embedding_parameter_count": update_summary["non_embedding_parameter_count"],
        "updated_non_embedding_parameter_count": update_summary["updated_non_embedding_parameter_count"],
        "non_embedding_update_fraction": update_summary["non_embedding_update_fraction"],
        "trainable_tensor_count": update_summary["trainable_tensor_count"],
        "updated_tensor_count": update_summary["updated_tensor_count"],
        "parameter_tensor_update_fraction": update_summary["parameter_tensor_update_fraction"],
        "non_embedding_tensor_update_fraction": update_summary["non_embedding_tensor_update_fraction"],
        "train_steps": steps,
        "optimizer_step_count": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "window_size": seq_len,
        "stride": 1,
        "source_train_token_count": len(train_tokens),
        "source_eval_token_count": len(eval_tokens),
        "train_window_count": train_window_count,
        "eval_window_count": eval_window_count,
        "windows_consumed": windows_consumed,
        "token_positions_consumed": token_positions_consumed,
        "train_token_positions_seen": token_positions_consumed,
        "train_corpus_token_coverage_fraction": round(token_positions_consumed / max(1, len(train_tokens)), 6),
        "learning_rate": lr,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_curve": train_loss_curve,
        "train_loss_curve_sampled": len(train_loss_curve) != len(losses),
        "train_loss_curve_point_count": len(losses),
        "eval_loss_before": eval_loss_before,
        "eval_loss_after": eval_loss,
        "eval_perplexity_before": round(math.exp(min(20.0, eval_loss_before)), 6) if math.isfinite(eval_loss_before) else math.inf,
        "eval_perplexity_after": round(math.exp(min(20.0, eval_loss)), 6) if math.isfinite(eval_loss) else math.inf,
        "heldout_lm_loss_curve": [eval_loss_before, eval_loss],
        "heldout_lm_improved": bool(math.isfinite(eval_loss_before) and math.isfinite(eval_loss) and eval_loss < eval_loss_before),
        "eval_loss": eval_loss,
        "eval_perplexity": round(math.exp(min(20.0, eval_loss)), 6) if math.isfinite(eval_loss) else math.inf,
        "wall_time_ms": int((time.perf_counter() - started) * 1000),
        "device": str(device),
        "from_scratch": True,
        "open_or_pretrained_model_weights_used": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit_count": 0,
    }


class TransformerLm:
    def __new__(cls, *args: Any, torch: Any, **kwargs: Any) -> Any:
        nn = torch.nn

        class _Model(nn.Module):
            def __init__(self, vocab_size: int, *, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, max_len: int) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
                self.position = nn.Parameter(torch.zeros(1, max_len, d_model))
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
                self.output = nn.Linear(d_model, vocab_size)

            def forward(self, x: Any) -> Any:
                h = self.embedding(x) + self.position[:, : x.shape[1], :]
                mask = torch.triu(torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool, device=x.device), diagonal=1)
                return self.output(self.encoder(h, mask=mask))

        return _Model(*args, **kwargs)


class SymLiquidLm:
    def __new__(cls, *args: Any, torch: Any, **kwargs: Any) -> Any:
        nn = torch.nn

        class _Model(nn.Module):
            def __init__(self, vocab_size: int, *, hidden_dim: int, reservoir_dim: int, hv_dim: int) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
                self.liquid_in = nn.Linear(hidden_dim, hidden_dim)
                self.liquid_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.tau = nn.Linear(hidden_dim, hidden_dim)
                self.reservoir = nn.Linear(hidden_dim, reservoir_dim)
                self.vsa = nn.Linear(reservoir_dim, hv_dim, bias=False)
                self.context = nn.Linear(hv_dim, hidden_dim)
                self.output = nn.Linear(hidden_dim, vocab_size)

            def forward(self, x: Any) -> Any:
                emb = self.embedding(x)
                h = emb.new_zeros((x.shape[0], self.liquid_h.out_features))
                memory = emb.new_zeros((x.shape[0], self.vsa.out_features))
                outs = []
                for t in range(x.shape[1]):
                    xt = emb[:, t, :]
                    candidate = torch.tanh(self.liquid_in(xt) + self.liquid_h(h))
                    alpha = torch.sigmoid(self.tau(xt))
                    h = (1.0 - alpha) * h + alpha * candidate
                    hv = torch.tanh(self.vsa(torch.tanh(self.reservoir(h))))
                    memory = 0.97 * memory + hv
                    ctx = torch.tanh(self.context(memory / memory.norm(dim=-1, keepdim=True).clamp(min=1e-6)))
                    outs.append(self.output(ctx).unsqueeze(1))
                return torch.cat(outs, dim=1)

        return _Model(*args, **kwargs)


def sample_lm_batch(tokens: list[int], seq_len: int, batch_size: int, *, torch: Any, device: Any) -> tuple[Any, Any]:
    max_start = max(1, len(tokens) - seq_len - 1)
    starts = [random.randrange(0, max_start) for _ in range(batch_size)]
    x = [tokens[start : start + seq_len] for start in starts]
    y = [tokens[start + 1 : start + seq_len + 1] for start in starts]
    return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)


def evaluate_lm(model: Any, tokens: list[int], seq_len: int, batch_size: int, *, torch: Any, device: Any) -> float:
    if len(tokens) < seq_len + 2:
        return math.nan
    losses = []
    model.eval()
    with torch.no_grad():
        for _ in range(min(16, max(1, len(tokens) // max(1, seq_len)))):
            x, y = sample_lm_batch(tokens, seq_len, batch_size, torch=torch, device=device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=0)
            losses.append(float(loss.detach().cpu()))
    model.train()
    return round(sum(losses) / max(1, len(losses)), 6)


def compact_loss_curve(losses: list[float], max_points: int) -> list[float]:
    if max_points <= 0 or len(losses) <= max_points:
        return list(losses)
    if max_points <= 2:
        return [losses[0], losses[-1]][:max_points]
    stride = max(1, len(losses) // max(1, max_points - 1))
    compact = [losses[idx] for idx in range(0, len(losses), stride)][: max_points - 1]
    if compact[-1] != losses[-1]:
        compact.append(losses[-1])
    return compact[:max_points]


def model_parameter_snapshot(model: Any, *, torch: Any) -> dict[str, Any]:
    return {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
        if bool(getattr(param, "requires_grad", False))
    }


def model_parameter_update_summary(model: Any, before: dict[str, Any], *, torch: Any) -> dict[str, Any]:
    changed = 0
    total = 0
    changed_non_embedding = 0
    total_non_embedding = 0
    changed_tensors = 0
    tensor_total = 0
    changed_non_embedding_tensors = 0
    non_embedding_tensor_total = 0
    eps = 1e-12
    for name, param in model.named_parameters():
        if not bool(getattr(param, "requires_grad", False)) or name not in before:
            continue
        current = param.detach().cpu()
        delta = (current - before[name]).abs()
        changed_elements = int(delta.gt(eps).sum().item())
        count = int(param.numel())
        total += count
        changed += changed_elements
        tensor_total += 1
        if changed_elements > 0:
            changed_tensors += 1
        if "embedding" not in str(name):
            total_non_embedding += count
            changed_non_embedding += changed_elements
            non_embedding_tensor_total += 1
            if changed_elements > 0:
                changed_non_embedding_tensors += 1
    return {
        "trainable_parameter_count": total,
        "updated_parameter_count": changed,
        "parameter_update_fraction": ratio(changed, total),
        "non_embedding_parameter_count": total_non_embedding,
        "updated_non_embedding_parameter_count": changed_non_embedding,
        "non_embedding_update_fraction": ratio(changed_non_embedding, total_non_embedding),
        "trainable_tensor_count": tensor_total,
        "updated_tensor_count": changed_tensors,
        "parameter_tensor_update_fraction": ratio(changed_tensors, tensor_total),
        "non_embedding_tensor_count": non_embedding_tensor_total,
        "updated_non_embedding_tensor_count": changed_non_embedding_tensors,
        "non_embedding_tensor_update_fraction": ratio(changed_non_embedding_tensors, non_embedding_tensor_total),
    }


def build_gates(admissions: list[dict[str, Any]], tokenizer: BpeTokenizer, train_tokens: list[int], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    admitted = [row for row in admissions if row["admitted"]]
    token_total = sum(int(row.get("rough_token_count") or 0) for row in admitted)
    algorithmic_fraction = algorithmic_python_token_fraction(admitted)
    completed = [row for row in runs if row.get("ok")]
    max_token_positions = max([int(row.get("token_positions_consumed") or 0) for row in completed] or [0])
    max_window_consumption = max([int(row.get("windows_consumed") or 0) for row in completed] or [0])
    transformer_runs = [row for row in completed if str(row.get("arm_id")) == "transformer_control"]
    transformer_update_ok = any(
        float(row.get("parameter_update_fraction") or 0.0) >= 0.98
        and float(row.get("non_embedding_update_fraction") or 0.0) >= 0.90
        for row in transformer_runs
    )
    heldout_improved = any(bool(row.get("heldout_lm_improved")) for row in transformer_runs)
    return [
        gate("corpus_documents_admitted", len(admitted) > 0, len(admitted), "hard"),
        gate("public_benchmark_payload_admitted_zero", not any(row["public_benchmark_payload_detected"] for row in admitted), 0, "hard"),
        gate("eval_overlap_admitted_zero", not any(row.get("eval_overlap_detected") for row in admitted), 0, "hard"),
        gate("all_admitted_sources_license_allowed", all(row.get("license_allowed") for row in admitted), value_breakdown(admitted, "license"), "hard"),
        gate("rough_tokens_materially_above_old_80k_canary", token_total >= 1_000_000, token_total, "warning"),
        gate("algorithmic_python_fraction_nontrivial", algorithmic_fraction >= 0.50, algorithmic_fraction, "warning"),
        gate("tokenizer_vocab_above_canary_floor", len(tokenizer.vocab) >= 256, len(tokenizer.vocab), "hard"),
        gate("train_tokens_above_canary_floor", len(train_tokens) >= 1024, len(train_tokens), "hard"),
        gate("from_scratch_pretraining_runs_completed", len(completed) >= 1, len(runs), "hard"),
        gate("transformer_survival_lane_completed", bool(transformer_runs), len(transformer_runs), "hard"),
        gate("million_scale_optimizer_token_positions_consumed", max_token_positions >= 1_000_000, max_token_positions, "warning"),
        gate("optimizer_windows_consumed_reported", max_window_consumption > 0, max_window_consumption, "hard"),
        gate("transformer_full_parameter_update_floor", transformer_update_ok, {
            "max_parameter_update_fraction": max([float(row.get("parameter_update_fraction") or 0.0) for row in transformer_runs] or [0.0]),
            "max_non_embedding_update_fraction": max([float(row.get("non_embedding_update_fraction") or 0.0) for row in transformer_runs] or [0.0]),
        }, "hard"),
        gate("transformer_heldout_lm_improved", heldout_improved, [
            {
                "budget_id": row.get("budget_id"),
                "eval_loss_before": row.get("eval_loss_before"),
                "eval_loss_after": row.get("eval_loss_after"),
                "heldout_lm_improved": row.get("heldout_lm_improved"),
            }
            for row in transformer_runs
        ], "warning"),
        gate("open_or_pretrained_model_weights_used_false", not any(row.get("open_or_pretrained_model_weights_used") for row in runs), False, "hard"),
        gate("external_inference_calls_zero", sum(int(row.get("external_inference_calls") or 0) for row in runs) == 0, 0, "hard"),
        gate("pretraining_budget_present", len({row.get("budget_id") for row in runs if row.get("ok")}) >= 1, sorted({row.get("budget_id") for row in runs}), "warning"),
    ]


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence, "severity": severity}


def ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator or 0.0)
    if denominator == 0.0:
        return 0.0
    return round(float(numerator or 0.0) / denominator, 6)


def summarize_admissions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(str(row.get("reason")) for row in rows)
    admitted = [row for row in rows if row["admitted"]]
    return {
        "source_count": len(rows),
        "admitted_document_count": len(admitted),
        "admitted_rough_token_count": sum(int(row.get("rough_token_count") or 0) for row in admitted),
        "content_token_breakdown": content_token_breakdown(admitted),
        "content_document_breakdown": value_breakdown(admitted, "content_type"),
        "license_breakdown": value_breakdown(admitted, "license"),
        "source_kind_breakdown": value_breakdown(admitted, "source_kind"),
        "algorithmic_python_token_fraction": algorithmic_python_token_fraction(admitted),
        "rejected_reason_counts": dict(sorted(reasons.items())),
        "admitted_paths_sample": [row["path"] for row in admitted[:24]],
        "public_reference_token_counts": dict(sorted(Counter(token for row in rows for token in row.get("public_reference_tokens", [])).items())),
    }


def value_breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "unknown") for row in rows).items()))


def content_token_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get("content_type") or "unknown")] += int(row.get("rough_token_count") or 0)
    return dict(sorted(counts.items()))


def algorithmic_python_token_fraction(rows: list[dict[str, Any]]) -> float:
    total = sum(int(row.get("rough_token_count") or 0) for row in rows)
    if total <= 0:
        return 0.0
    algorithmic = sum(int(row.get("rough_token_count") or 0) for row in rows if row.get("algorithmic_python"))
    return round(algorithmic / total, 6)


def tokenizer_summary(tokenizer: BpeTokenizer) -> dict[str, Any]:
    return {
        "policy": "project_theseus_bpe_tokenizer_v1",
        "vocab_size": len(tokenizer.vocab),
        "merge_count": len(tokenizer.merges),
        "special_tokens": SPECIAL_TOKENS,
        "from_scratch": True,
    }


def public_reference_tokens(path: str, text: str) -> set[str]:
    haystack = f"{path}\n{text[:20000]}".lower()
    return {token for token in PUBLIC_BENCHMARK_TOKENS if token in haystack}


def public_payload_detected(path: str, text: str, refs: set[str]) -> bool:
    lower_path = path.lower()
    lower_text = text[:50000].lower()
    if any(token in lower_path for token in PUBLIC_BENCHMARK_TOKENS):
        return True
    if not refs:
        return False
    if any(hint in lower_text for hint in PUBLIC_PAYLOAD_HINTS) and path.endswith((".json", ".jsonl", ".py", ".md")):
        return True
    return False


def rough_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\w\s]", text))


def choose_torch_device(torch: Any, preference: str) -> Any:
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # CPU is more stable for small canaries because the strict comparator
        # already keeps MPS off for transformer-mask compatibility.
    return torch.device("cpu")


def count_params(model: Any) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def default_roots() -> list[str]:
    return ["README.md", "AGENTS.md", "docs", "scripts", "crates", "configs"]


def default_suffixes() -> list[str]:
    return [".md", ".py", ".rs", ".toml", ".json", ".sh"]


def default_excluded_dirs() -> list[str]:
    return [
        ".git",
        "__pycache__",
        ".venv",
        ".venv-mlx",
        "target",
        "reports",
        "archive",
        "deprecated",
        "checkpoints",
        "dist",
        "data",
        "resource_pantry",
        "benchmarks",
        "tmp",
    ]


def default_budgets() -> list[dict[str, Any]]:
    return [
        {"id": "tiny_canary", "seq_len": 48, "steps": 24, "batch_size": 12, "d_model": 48, "learning_rate": 0.003},
        {"id": "small_canary", "seq_len": 64, "steps": 36, "batch_size": 12, "d_model": 64, "learning_rate": 0.002},
    ]


def default_config() -> dict[str, Any]:
    return {
        "policy": "project_theseus_narrow_corpus_pretraining_spine_config_v1",
        "scope": {
            "include": ["English prose", "Python/code", "Project Theseus docs/source/configs"],
            "exclude": ["public benchmark payloads", "open/base/pretrained weights", "uncertain-license data"],
        },
        "corpus": {
            "roots": default_roots(),
            "suffixes": default_suffixes(),
            "excluded_dirs": default_excluded_dirs(),
            "max_files": 384,
            "max_chars_per_file": 30_000,
            "max_total_chars": 900_000,
            "project_internal_license": "project-internal",
        },
        "tokenizer": {
            "target_vocab_size": 8192,
            "min_vocab_size": 256,
            "min_pair_frequency": 2,
            "max_merges": 8192,
            "max_train_chars": 900_000,
        },
        "pretraining": {
            "seed": 23,
            "device": "auto",
            "eval_fraction": 0.10,
            "max_total_tokens": 180_000,
            "budgets": default_budgets(),
        },
    }


def load_config(path: Path) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return default_config()


def strip_tensor_free(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_tensor_free(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_tensor_free(item) for item in value]
    return value


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
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


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
