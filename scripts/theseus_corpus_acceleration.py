#!/usr/bin/env python3
"""Qualify the independent Rust exact corpus-to-tensor implementation.

The Python production path remains the parity oracle. This script samples the
already-admitted canonical index, never emits source text in its report, and
requires byte-identical input, label, and mask arrays before reporting speed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from moecot_language_tokenizer import encode_document
from moecot_source_conditioned_pretraining import (
    encode_kerc_global_target as python_encode_kerc_global_target,
    kerc_code_space,
    kerc_code_tokens,
)


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = (
    "english_conversation_instruction",
    "english_broad",
    "python",
    "javascript_typescript",
    "html_css",
    "rust",
)
ARRAY_NAMES = {
    "inputs": "canonical_pretrain_inputs_v1.i32",
    "labels": "canonical_pretrain_labels_v1.i32",
    "mask": "canonical_pretrain_mask_v1.u8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("python", "kerc-python"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "runtime/standard_causal_transformer_scale_v2/stage_metadata_v1.json",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT
        / "runtime/standard_causal_transformer_scale_v2/canonical_pretrain_index_v2.sqlite3",
    )
    parser.add_argument(
        "--rust-bin", type=Path, default=ROOT / "target/release/theseus-corpus"
    )
    parser.add_argument("--sample-per-category", type=int, default=96)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "reports/theseus_corpus_acceleration.json"
    )
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-offset", type=int)
    parser.add_argument(
        "--kerc-corpus",
        type=Path,
        default=ROOT / "data/training_data/moecot_kernel_english_v1/private_train.jsonl",
    )
    parser.add_argument(
        "--kerc-vocab",
        type=Path,
        default=ROOT / "data/training_data/moecot_kernel_english_v1/code_vocabulary_v1.json",
    )
    parser.add_argument("--kerc-sample-count", type=int, default=500)
    parser.add_argument("--kernel-offset", type=int)
    parser.add_argument("--pointer-offset", type=int)
    parser.add_argument("--skip-kerc", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker == "python":
        if not args.sample or not args.vocab or not args.output_dir:
            raise ValueError("python worker requires sample, vocab, and output-dir")
        if args.target_offset is None:
            raise ValueError("python worker requires target-offset")
        print(
            json.dumps(
                python_materialize(
                    args.sample,
                    args.vocab,
                    args.output_dir,
                    target_offset=args.target_offset,
                    sequence_length=args.sequence_length,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.worker == "kerc-python":
        if not args.sample or not args.vocab:
            raise ValueError("KERC Python worker requires sample and vocab")
        if args.kernel_offset is None or args.pointer_offset is None:
            raise ValueError("KERC Python worker requires both typed offsets")
        print(
            json.dumps(
                python_kerc_benchmark(
                    args.sample,
                    args.vocab,
                    kernel_offset=args.kernel_offset,
                    pointer_offset=args.pointer_offset,
                ),
                sort_keys=True,
            )
        )
        return 0
    report = qualify(args)
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["state"] == "GREEN" else 2


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_per_category <= 0 or args.repetitions <= 0:
        raise ValueError("sample count and repetitions must be positive")
    metadata = read_json(args.metadata)
    target_vocab = {str(key): int(value) for key, value in metadata["target_vocab"].items()}
    source_vocab = metadata.get("source_vocab") or {}
    target_offset = 3 + len(source_vocab)
    if not args.rust_bin.is_file():
        raise FileNotFoundError(f"release Rust corpus binary missing: {args.rust_bin}")

    with tempfile.TemporaryDirectory(prefix="theseus-corpus-qualification-") as raw_work:
        work = Path(raw_work)
        sample_path = work / "sample.jsonl"
        vocab_path = work / "target_vocab.json"
        sample = export_sample(args.index, sample_path, args.sample_per_category)
        vocab_path.write_text(json.dumps({"target_vocab": target_vocab}), encoding="utf-8")

        python_runs = []
        rust_runs = []
        for index in range(args.repetitions):
            python_output = work / f"python-{index}"
            rust_output = work / f"rust-{index}"
            python_runs.append(
                run_measured(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "python",
                        "--sample",
                        str(sample_path),
                        "--vocab",
                        str(vocab_path),
                        "--output-dir",
                        str(python_output),
                        "--target-offset",
                        str(target_offset),
                        "--sequence-length",
                        str(args.sequence_length),
                    ]
                )
            )
            rust_runs.append(
                run_measured(
                    [
                        str(args.rust_bin),
                        "materialize",
                        "--vocab",
                        str(vocab_path),
                        "--input",
                        str(sample_path),
                        "--output-dir",
                        str(rust_output),
                        "--target-offset",
                        str(target_offset),
                        "--sequence-length",
                        str(args.sequence_length),
                    ]
                )
            )

        parity = compare_runs(python_runs, rust_runs)
        restart = compare_restart_identity(rust_runs)
        ingestion = qualify_ingestion_and_parallelism(
            args,
            work,
            sample_path=sample_path,
            vocab_path=vocab_path,
            target_offset=target_offset,
            reference_run=rust_runs[0],
        )
        python_wall = [float(row["wall_seconds"]) for row in python_runs]
        rust_wall = [float(row["wall_seconds"]) for row in rust_runs]
        speedup = statistics.median(python_wall) / statistics.median(rust_wall)
        python_peak = max(int(row["peak_rss_bytes"]) for row in python_runs)
        rust_peak = max(int(row["peak_rss_bytes"]) for row in rust_runs)
        memory_not_increased = rust_peak <= python_peak
        exact_state = (
            "GREEN"
            if parity["fault_count"] == 0
            and restart["exact_restart"]
            and speedup >= 5.0
            and memory_not_increased
            else "YELLOW"
        )
        kerc = (
            {"state": "SKIPPED", "reason": "explicit_skip_kerc"}
            if args.skip_kerc
            else qualify_kerc(
                args,
                work,
                kernel_offset=target_offset + len(target_vocab),
            )
        )
        state = (
            "GREEN"
            if exact_state == "GREEN"
            and ingestion.get("state") == "GREEN"
            and kerc.get("state") in {"GREEN", "SKIPPED"}
            else "YELLOW"
        )
        return {
            "policy": "project_theseus_independent_corpus_acceleration_qualification_v1",
            "state": state,
            "implementation": "theseus-corpus Rust exact-text corpus-to-tensor",
            "reference": "canonical Python exact-text corpus-to-tensor",
            "sample": sample,
            "frozen_abi": {
                "stage_metadata_path": str(args.metadata),
                "stage_metadata_sha256": file_sha256(args.metadata),
                "stage_signature": metadata.get("stage_signature"),
                "index_path": str(args.index),
                "index_sha256": file_sha256(args.index),
                "target_vocab_size": len(target_vocab),
                "source_vocab_size": len(source_vocab),
                "target_offset": target_offset,
                "sequence_length": args.sequence_length,
                "row_order": "category_then_digest_sample; source order preserved by both routes",
            },
            "parity": parity,
            "restart": restart,
            "exact_corpus_to_tensor_state": exact_state,
            "performance": {
                "python_runs": redact_runs(python_runs),
                "rust_runs": redact_runs(rust_runs),
                "python_median_wall_seconds": statistics.median(python_wall),
                "rust_median_wall_seconds": statistics.median(rust_wall),
                "corpus_to_tensor_speedup": speedup,
                "python_peak_rss_bytes": python_peak,
                "rust_peak_rss_bytes": rust_peak,
                "peak_memory_not_increased": memory_not_increased,
                "acceptance_speedup": 5.0,
            },
            "ingestion_parallelism_and_scanner": ingestion,
            "kerc_dual_space": kerc,
            "route_adoption_ready": state == "GREEN",
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "claim_boundary": (
                "Representative frozen-lineage corpus-to-tensor and typed KERC mechanics only; "
                "this is not end-to-end training speed, model quality, or cross-platform evidence."
            ),
        }


def qualify_ingestion_and_parallelism(
    args: argparse.Namespace,
    work: Path,
    *,
    sample_path: Path,
    vocab_path: Path,
    target_offset: int,
    reference_run: dict[str, Any],
) -> dict[str, Any]:
    import zstandard

    raw = sample_path.read_bytes()
    gzip_path = work / "sample.jsonl.gz"
    gzip_path.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
    zstd_path = work / "sample.jsonl.zst"
    zstd_path.write_bytes(zstandard.ZstdCompressor(level=3).compress(raw))

    materializations: dict[str, dict[str, Any]] = {}
    cases = (
        ("plain_single", sample_path, 1),
        ("plain_parallel", sample_path, min(4, os.cpu_count() or 1)),
        ("gzip_parallel", gzip_path, min(4, os.cpu_count() or 1)),
        ("zstd_parallel", zstd_path, min(4, os.cpu_count() or 1)),
    )
    for name, source, workers in cases:
        output = work / f"ingestion-{name}"
        materializations[name] = run_measured(
            [
                str(args.rust_bin),
                "materialize",
                "--vocab",
                str(vocab_path),
                "--input",
                str(source),
                "--output-dir",
                str(output),
                "--target-offset",
                str(target_offset),
                "--sequence-length",
                str(args.sequence_length),
                "--workers",
                str(workers),
                "--chunk-rows",
                "64",
            ]
        )
    parity_faults = []
    for name, candidate in materializations.items():
        compared = compare_runs([reference_run], [candidate])
        if compared["fault_count"]:
            parity_faults.append({"case": name, "faults": compared["faults"]})

    scanner_trials: list[dict[str, Any]] = []
    for trial_index in range(3):
        reports: dict[str, dict[str, Any]] = {}
        order = (
            ("scalar", "accelerated")
            if trial_index % 2 == 0
            else ("accelerated", "scalar")
        )
        for mode in order:
            command = [
                str(args.rust_bin),
                "scanner-benchmark",
                "--input",
                str(sample_path),
                "--repetitions",
                str(args.repetitions),
            ]
            if mode == "scalar":
                command.append("--scalar")
            reports[mode] = run_measured(command)
        scalar_trial_median = statistics.median(
            float(row["seconds"]) for row in reports["scalar"]["runs"]
        )
        accelerated_trial_median = statistics.median(
            float(row["seconds"]) for row in reports["accelerated"]["runs"]
        )
        scanner_trials.append(
            {
                "trial": trial_index + 1,
                "order": list(order),
                "scalar_median_seconds": scalar_trial_median,
                "accelerated_median_seconds": accelerated_trial_median,
                "speedup": scalar_trial_median
                / max(accelerated_trial_median, sys.float_info.min),
                "scalar": reports["scalar"],
                "accelerated": reports["accelerated"],
            }
        )
    scalar_runs = [
        row
        for trial in scanner_trials
        for row in trial["scalar"]["runs"]
    ]
    accelerated_runs = [
        row
        for trial in scanner_trials
        for row in trial["accelerated"]["runs"]
    ]
    scanner_digest_parity = {
        str(row["output_digest"]) for row in scalar_runs
    } == {str(row["output_digest"]) for row in accelerated_runs}
    scalar_median = statistics.median(float(row["seconds"]) for row in scalar_runs)
    accelerated_median = statistics.median(
        float(row["seconds"]) for row in accelerated_runs
    )
    scanner_trial_speedups = [float(row["speedup"]) for row in scanner_trials]
    scanner_speedup = statistics.median(scanner_trial_speedups)
    scanner_minimum_speedup = min(scanner_trial_speedups)
    single_seconds = float(materializations["plain_single"]["wall_seconds"])
    parallel_seconds = float(materializations["plain_parallel"]["wall_seconds"])
    parallel_speedup = single_seconds / max(parallel_seconds, sys.float_info.min)
    # A few percent is within normal process-launch and thermal noise on this
    # host. Keep the scalar oracle unless the specialized scanner clears a
    # material 10% wall-time win under exact digest parity.
    scanner_adoption_speedup = 1.10
    scanner_disposition = (
        "ADOPTED"
        if scanner_digest_parity
        and scanner_speedup >= scanner_adoption_speedup
        and scanner_minimum_speedup >= 1.0
        else "REJECTED"
    )
    parallel_disposition = "ADOPTED" if parallel_speedup >= 1.05 else "REJECTED"
    cache_root = work / "content-cache"
    cache_command = [
        str(args.rust_bin),
        "cache-materialize",
        "--vocab",
        str(vocab_path),
        "--input",
        str(sample_path),
        "--cache-root",
        str(cache_root),
        "--target-offset",
        str(target_offset),
        "--sequence-length",
        str(args.sequence_length),
        "--workers",
        str(min(4, os.cpu_count() or 1)),
        "--chunk-rows",
        "64",
        "--cache-budget-bytes",
        str(512 * 1024 * 1024),
    ]
    cache_cold = run_measured(cache_command)
    cache_warm = run_measured(cache_command)
    cache_identity_exact = bool(
        cache_cold.get("cache_hit") is False
        and cache_warm.get("cache_hit") is True
        and cache_cold.get("materialization_identity_sha256")
        == cache_warm.get("materialization_identity_sha256")
        and cache_cold.get("manifest_sha256") == cache_warm.get("manifest_sha256")
    )
    return {
        "policy": "project_theseus_corpus_ingestion_parallel_scanner_qualification_v1",
        "state": (
            "GREEN"
            if not parity_faults and scanner_digest_parity and cache_identity_exact
            else "RED"
        ),
        "compressed_input": {
            "codecs": ["plain", "gzip", "zstd"],
            "exact_artifact_parity": not parity_faults,
            "faults": parity_faults,
            "raw_bytes": len(raw),
            "gzip_bytes": gzip_path.stat().st_size,
            "zstd_bytes": zstd_path.stat().st_size,
        },
        "bounded_parallel_materialization": {
            "workers": materializations["plain_parallel"].get("worker_count"),
            "chunk_rows": materializations["plain_parallel"].get("chunk_rows"),
            "exact_row_and_artifact_identity": not parity_faults,
            "single_wall_seconds": single_seconds,
            "parallel_wall_seconds": parallel_seconds,
            "speedup": parallel_speedup,
            "disposition": parallel_disposition,
            "reason": (
                "qualified_exact_and_materially_faster"
                if parallel_disposition == "ADOPTED"
                else "qualified_exact_but_not_faster_on_representative_sample"
            ),
        },
        "scanner": {
            "implementation": "bounded_aarch64_neon_ascii_runs_with_scalar_tail",
            "scalar_median_seconds": scalar_median,
            "accelerated_median_seconds": accelerated_median,
            "speedup": scanner_speedup,
            "minimum_process_pair_speedup": scanner_minimum_speedup,
            "process_pair_speedups": scanner_trial_speedups,
            "process_pair_count": len(scanner_trials),
            "adoption_speedup": scanner_adoption_speedup,
            "exact_digest_parity": scanner_digest_parity,
            "disposition": scanner_disposition,
            "reason": (
                "qualified_exact_and_materially_faster"
                if scanner_disposition == "ADOPTED"
                else "qualified_exact_but_not_faster_on_representative_sample"
            ),
        },
        "content_addressed_cache": {
            "state": "GREEN" if cache_identity_exact else "RED",
            "cold_hit": cache_cold.get("cache_hit"),
            "warm_hit": cache_warm.get("cache_hit"),
            "exact_identity_and_manifest_reuse": cache_identity_exact,
            "cold_process_wall_seconds": cache_cold.get("process_wall_seconds"),
            "warm_process_wall_seconds": cache_warm.get("process_wall_seconds"),
            "warm_speedup": float(cache_cold.get("process_wall_seconds") or 0.0)
            / max(float(cache_warm.get("process_wall_seconds") or 0.0), sys.float_info.min),
            "cache_budget_bytes": ((cache_warm.get("cache") or {}).get("budget_bytes")),
            "resident_bytes": ((cache_warm.get("cache") or {}).get("resident_bytes")),
            "corruption_rejection_covered_by_integration_test": True,
        },
        "materialization_runs": {
            name: {
                "input_codec": row.get("input_codec"),
                "worker_count": row.get("worker_count"),
                "chunk_rows": row.get("chunk_rows"),
                "wall_seconds": row.get("wall_seconds"),
                "peak_rss_bytes": row.get("peak_rss_bytes"),
                "artifacts": row.get("artifacts"),
            }
            for name, row in materializations.items()
        },
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "claim_boundary": "ingestion, scanner, and exact materialization mechanics only",
    }


def qualify_kerc(
    args: argparse.Namespace, work: Path, *, kernel_offset: int
) -> dict[str, Any]:
    if args.kerc_sample_count < 2:
        raise ValueError("KERC sample count must be at least two")
    if not args.kerc_corpus.is_file() or not args.kerc_vocab.is_file():
        return {
            "state": "MISSING",
            "reason": "canonical_kerc_corpus_or_vocabulary_missing",
            "corpus_exists": args.kerc_corpus.is_file(),
            "vocabulary_exists": args.kerc_vocab.is_file(),
        }
    code_vocab = read_json(args.kerc_vocab)
    pointer_offset = kernel_offset + len(code_vocab.get("kernel_vocab") or {})
    sample_path = work / "kerc-sample.jsonl"
    sample = export_kerc_sample(args.kerc_corpus, sample_path, args.kerc_sample_count)

    python_output = work / "kerc-python-parity.jsonl"
    rust_output = work / "kerc-rust-parity.jsonl"
    write_python_kerc_parity(
        sample_path,
        args.kerc_vocab,
        python_output,
        kernel_offset=kernel_offset,
        pointer_offset=pointer_offset,
    )
    subprocess.run(
        [
            str(args.rust_bin),
            "kerc-encode",
            "--code-vocab",
            str(args.kerc_vocab),
            "--input",
            str(sample_path),
            "--output",
            str(rust_output),
            "--kernel-offset",
            str(kernel_offset),
            "--pointer-offset",
            str(pointer_offset),
            "--include-tokens",
        ],
        cwd=ROOT,
        check=True,
    )
    parity = compare_kerc_parity(python_output, rust_output)

    python_runs = []
    rust_runs = []
    for _ in range(args.repetitions):
        python_runs.append(
            run_measured(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "kerc-python",
                    "--sample",
                    str(sample_path),
                    "--vocab",
                    str(args.kerc_vocab),
                    "--kernel-offset",
                    str(kernel_offset),
                    "--pointer-offset",
                    str(pointer_offset),
                ]
            )
        )
        rust = run_measured(
            [
                str(args.rust_bin),
                "kerc-benchmark",
                "--code-vocab",
                str(args.kerc_vocab),
                "--input",
                str(sample_path),
                "--kernel-offset",
                str(kernel_offset),
                "--pointer-offset",
                str(pointer_offset),
                "--repetitions",
                "1",
            ]
        )
        rust_measurement = (rust.get("runs") or [{}])[0]
        rust.update(rust_measurement)
        rust["wall_seconds"] = float(rust_measurement.get("seconds") or 0.0)
        rust_runs.append(rust)

    python_wall = [float(row["wall_seconds"]) for row in python_runs]
    rust_wall = [float(row["wall_seconds"]) for row in rust_runs]
    speedup = statistics.median(python_wall) / max(
        statistics.median(rust_wall), sys.float_info.min
    )
    digest_parity = all(
        reference.get("output_digest") == candidate.get("output_digest")
        for reference, candidate in zip(python_runs, rust_runs)
    )
    python_peak = max(int(row["peak_rss_bytes"]) for row in python_runs)
    rust_peak = max(int(row["peak_rss_bytes"]) for row in rust_runs)
    memory_not_increased = rust_peak <= python_peak
    state = (
        "GREEN"
        if parity["fault_count"] == 0
        and digest_parity
        and speedup >= 5.0
        and memory_not_increased
        else "YELLOW"
    )
    return {
        "policy": "project_theseus_kerc_dual_space_acceleration_qualification_v1",
        "state": state,
        "sample": sample,
        "vocabulary": {
            "path": str(args.kerc_vocab),
            "sha256": file_sha256(args.kerc_vocab),
            "contract_sha256": code_vocab.get("contract_sha256"),
            "kernel_vocab_size": len(code_vocab.get("kernel_vocab") or {}),
            "pointer_vocab_size": len(code_vocab.get("pointer_vocab") or {}),
            "kernel_offset": kernel_offset,
            "pointer_offset": pointer_offset,
        },
        "parity": {**parity, "aggregate_output_digest_parity": digest_parity},
        "performance": {
            "python_runs": redact_kerc_runs(python_runs),
            "rust_runs": redact_kerc_runs(rust_runs),
            "python_median_wall_seconds": statistics.median(python_wall),
            "rust_median_wall_seconds": statistics.median(rust_wall),
            "dual_space_encoding_speedup": speedup,
            "python_peak_rss_bytes": python_peak,
            "rust_peak_rss_bytes": rust_peak,
            "peak_memory_not_increased": memory_not_increased,
            "acceptance_speedup": 5.0,
        },
        "route_adoption_ready": state == "GREEN",
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "claim_boundary": "typed KERC tokenization and global dual-space encoding only",
    }


def export_kerc_sample(corpus: Path, output: Path, count: int) -> dict[str, Any]:
    objectives = (
        "surface_to_kernel_program_v1",
        "kernel_program_to_answer_packet_v1",
    )
    targets = {objectives[0]: count // 2 + count % 2, objectives[1]: count // 2}
    observed = {objective: 0 for objective in objectives}
    input_bytes = 0
    identities = hashlib.sha256()
    with corpus.open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as sink:
        for raw in source:
            row = json.loads(raw)
            objective = str(row.get("objective") or "")
            if objective not in targets or observed[objective] >= targets[objective]:
                continue
            target = str(row.get("target") or "")
            row_id = str(row.get("row_id") or row.get("target_sha256") or "")
            if not row_id:
                raise ValueError("KERC sample row lacks stable identity")
            sink.write(
                json.dumps(
                    {"id": row_id, "objective": objective, "target": target},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            observed[objective] += 1
            input_bytes += len(target.encode("utf-8"))
            identities.update(objective.encode())
            identities.update(b":")
            identities.update(row_id.encode())
            identities.update(b"\n")
            if observed == targets:
                break
    if observed != targets:
        raise ValueError(f"insufficient KERC rows: {observed}")
    return {
        "rows": sum(observed.values()),
        "rows_by_objective": observed,
        "input_utf8_bytes": input_bytes,
        "identity_digest": identities.hexdigest(),
        "source_path": str(corpus),
        "source_sha256": file_sha256(corpus),
        "contains_target_text_in_report": False,
    }


def python_kerc_benchmark(
    sample: Path, vocab_path: Path, *, kernel_offset: int, pointer_offset: int
) -> dict[str, Any]:
    code_vocab = read_json(vocab_path)
    rows = input_bytes = encoded_tokens = fallback_tokens = 0
    digest = hashlib.sha256()
    started = time.perf_counter()
    with sample.open(encoding="utf-8") as source:
        for raw in source:
            row = json.loads(raw)
            ids, receipt = python_encode_kerc_global_target(
                row["target"],
                code_vocabulary=code_vocab,
                kernel_offset=kernel_offset,
                pointer_offset=pointer_offset,
            )
            if int(receipt.get("unknown_token_count") or 0):
                raise ValueError(f"unrepresentable KERC row: {row['id']}")
            rows += 1
            input_bytes += len(row["target"].encode("utf-8"))
            encoded_tokens += len(ids)
            fallback_tokens += int(receipt.get("fallback_token_count") or 0)
            digest.update(row["id"].encode())
            for token_id in ids:
                digest.update(int(token_id).to_bytes(4, "little", signed=True))
    seconds = time.perf_counter() - started
    return {
        "policy": "project_theseus_kerc_dual_space_encoding_python_benchmark_v1",
        "rows": rows,
        "input_bytes": input_bytes,
        "encoded_tokens": encoded_tokens,
        "fallback_token_count": fallback_tokens,
        "wall_seconds": seconds,
        "mib_per_second": input_bytes / (1024 * 1024) / max(seconds, sys.float_info.min),
        "encoded_tokens_per_second": encoded_tokens / max(seconds, sys.float_info.min),
        "output_digest": digest.hexdigest(),
    }


def write_python_kerc_parity(
    sample: Path,
    vocab_path: Path,
    output: Path,
    *,
    kernel_offset: int,
    pointer_offset: int,
) -> None:
    code_vocab = read_json(vocab_path)
    with sample.open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as sink:
        for raw in source:
            row = json.loads(raw)
            tokens = kerc_code_tokens(row["target"])
            ids, receipt = python_encode_kerc_global_target(
                row["target"],
                code_vocabulary=code_vocab,
                kernel_offset=kernel_offset,
                pointer_offset=pointer_offset,
            )
            sink.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "objective": row["objective"],
                        "target_sha256": hashlib.sha256(row["target"].encode()).hexdigest(),
                        "ids_sha256": ids_sha256_python(ids),
                        "ids": ids,
                        "tokens": [
                            {"text": str(token), "space": kerc_code_space(token)}
                            for token in tokens
                        ],
                        "receipt": {
                            "encoded_token_count": int(receipt.get("encoded_token_count") or 0),
                            "encoded_tokens_by_space": receipt.get("encoded_tokens_by_space"),
                            "fallback_token_count": int(receipt.get("fallback_token_count") or 0),
                            "unknown_token_count": int(receipt.get("unknown_token_count") or 0),
                        },
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def compare_kerc_parity(reference: Path, candidate: Path) -> dict[str, Any]:
    faults = []
    rows = 0
    with reference.open(encoding="utf-8") as expected, candidate.open(encoding="utf-8") as actual:
        while True:
            left_raw = expected.readline()
            right_raw = actual.readline()
            if not left_raw and not right_raw:
                break
            rows += 1
            if not left_raw or not right_raw:
                faults.append({"row": rows, "field": "row_count"})
                break
            left = json.loads(left_raw)
            right = json.loads(right_raw)
            for field in ("id", "objective", "target_sha256", "ids_sha256", "ids", "tokens"):
                if left.get(field) != right.get(field):
                    faults.append({"row": rows, "field": field})
            for field in (
                "encoded_token_count",
                "encoded_tokens_by_space",
                "fallback_token_count",
                "unknown_token_count",
            ):
                if (left.get("receipt") or {}).get(field) != (right.get("receipt") or {}).get(field):
                    faults.append({"row": rows, "field": f"receipt.{field}"})
    return {
        "state": "GREEN" if not faults else "RED",
        "row_count": rows,
        "fault_count": len(faults),
        "faults": faults[:100],
        "exact_json_atom_code_space_id_and_receipt_identity": not faults,
    }


def ids_sha256_python(ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=True))
    return digest.hexdigest()


def redact_kerc_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "rows",
        "input_bytes",
        "encoded_tokens",
        "fallback_token_count",
        "wall_seconds",
        "process_wall_seconds",
        "mib_per_second",
        "encoded_tokens_per_second",
        "output_digest",
        "peak_rss_bytes",
    )
    return [{field: row.get(field) for field in fields} for row in runs]


def export_sample(index_path: Path, output: Path, per_category: int) -> dict[str, Any]:
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    counts: dict[str, int] = {}
    input_bytes = 0
    digest = hashlib.sha256()
    with sqlite3.connect(index_path) as connection, output.open("w", encoding="utf-8") as sink:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(documents)")
        }
        record_expression = "record_kind" if "record_kind" in columns else "'jsonl'"
        for category in CATEGORIES:
            rows = connection.execute(
                f"SELECT digest,path,byte_offset,byte_length,{record_expression} "
                "FROM documents WHERE category=? ORDER BY digest LIMIT ?",
                (category, per_category),
            ).fetchall()
            if len(rows) != per_category:
                raise ValueError(f"insufficient indexed rows for {category}: {len(rows)}")
            counts[category] = len(rows)
            for row_digest, path_value, offset, length, record_kind in rows:
                with Path(path_value).open("rb") as handle:
                    handle.seek(int(offset))
                    raw = handle.read(int(length))
                if str(record_kind) == "raw_text":
                    text = raw.decode("utf-8", errors="replace")
                else:
                    payload = json.loads(raw)
                    text = str(
                        payload.get("text")
                        if category in CATEGORIES[2:]
                        else payload.get("causal_text") or ""
                    )
                identity = f"{category}:{row_digest}"
                encoded = json.dumps(
                    {"id": identity, "category": category, "text": text},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                sink.write(encoded + "\n")
                input_bytes += len(text.encode("utf-8"))
                digest.update(category.encode())
                digest.update(b":")
                digest.update(str(row_digest).encode())
                digest.update(b"\n")
    return {
        "documents": sum(counts.values()),
        "documents_by_category": counts,
        "input_utf8_bytes": input_bytes,
        "identity_digest": digest.hexdigest(),
        "contains_source_text_in_report": False,
    }


def python_materialize(
    sample_path: Path,
    vocab_path: Path,
    output_dir: Path,
    *,
    target_offset: int,
    sequence_length: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    vocab = read_json(vocab_path)["target_vocab"]
    paths = {key: output_dir / name for key, name in ARRAY_NAMES.items()}
    started = time.perf_counter()
    documents = windows = positions = fallback_tokens = fallback_bytes = 0
    selected = hashlib.sha256()
    with paths["inputs"].open("wb") as inputs_sink, paths["labels"].open(
        "wb"
    ) as labels_sink, paths["mask"].open("wb") as mask_sink, sample_path.open(
        encoding="utf-8"
    ) as source:
        for raw in source:
            row = json.loads(raw)
            _tokens, local_ids, receipt = encode_document(
                row["text"], vocab, category=row["category"]
            )
            if int(receipt.get("unknown_token_count") or 0):
                raise ValueError(f"unrepresentable document: {row['id']}")
            if (receipt.get("roundtrip") or {}).get("state") != "GREEN":
                raise ValueError(f"roundtrip fault: {row['id']}")
            ids = [target_offset + int(value) for value in local_ids]
            fallback_tokens += int(receipt.get("fallback_token_count") or 0)
            fallback_bytes += int(receipt.get("fallback_byte_count") or 0)
            documents += 1
            selected.update(row["category"].encode())
            selected.update(b":")
            selected.update(row["id"].encode())
            selected.update(b"\n")
            for start in range(0, max(0, len(ids) - 1), sequence_length):
                width = min(sequence_length, len(ids) - start - 1)
                if width <= 0:
                    continue
                inputs = np.zeros((sequence_length,), dtype=np.int32)
                labels = np.zeros((sequence_length,), dtype=np.int32)
                mask = np.zeros((sequence_length,), dtype=np.uint8)
                inputs[:width] = ids[start : start + width]
                labels[:width] = ids[start + 1 : start + width + 1]
                mask[:width] = 1
                inputs.tofile(inputs_sink)
                labels.tofile(labels_sink)
                mask.tofile(mask_sink)
                windows += 1
                positions += width
    wall = time.perf_counter() - started
    return {
        "policy": "project_theseus_exact_corpus_to_tensor_python_reference_v1",
        "document_count": documents,
        "window_count": windows,
        "materialized_positions": positions,
        "sequence_length": sequence_length,
        "target_offset": target_offset,
        "fallback_token_count": fallback_tokens,
        "fallback_byte_count": fallback_bytes,
        "selected_document_digest": selected.hexdigest(),
        "wall_seconds": wall,
        "positions_per_second": positions / max(wall, sys.float_info.min),
        "artifacts": {
            key: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for key, path in paths.items()
        },
        "fallback_return_count": 0,
    }


def run_measured(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        ["/usr/bin/time", "-l", *command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"measured command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    payload = json.loads(result.stdout)
    rss_match = re.search(r"^\s*(\d+)\s+maximum resident set size$", result.stderr, re.MULTILINE)
    wall_match = re.search(r"^\s*([0-9.]+)\s+real\s+", result.stderr, re.MULTILINE)
    if not rss_match or not wall_match:
        raise ValueError("macOS time resource receipt was incomplete")
    payload["peak_rss_bytes"] = int(rss_match.group(1))
    payload["process_wall_seconds"] = float(wall_match.group(1))
    payload["wall_seconds"] = float(payload.get("wall_seconds") or wall_match.group(1))
    return payload


def compare_runs(
    python_runs: list[dict[str, Any]], rust_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(python_runs) != len(rust_runs):
        raise ValueError("reference and candidate run counts differ")
    faults = []
    for index, (reference, candidate) in enumerate(zip(python_runs, rust_runs)):
        for field in (
            "document_count",
            "window_count",
            "materialized_positions",
            "sequence_length",
            "target_offset",
            "fallback_token_count",
            "fallback_byte_count",
            "selected_document_digest",
        ):
            if reference.get(field) != candidate.get(field):
                faults.append(
                    {"run": index, "field": field, "expected": reference.get(field), "actual": candidate.get(field)}
                )
        for artifact in ARRAY_NAMES:
            expected = (reference.get("artifacts") or {}).get(artifact) or {}
            actual = (candidate.get("artifacts") or {}).get(artifact) or {}
            for field in ("bytes", "sha256"):
                if expected.get(field) != actual.get(field):
                    faults.append(
                        {
                            "run": index,
                            "artifact": artifact,
                            "field": field,
                            "expected": expected.get(field),
                            "actual": actual.get(field),
                        }
                    )
    return {
        "state": "GREEN" if not faults else "RED",
        "run_pairs": len(python_runs),
        "compared_fields_per_run": 14,
        "fault_count": len(faults),
        "faults": faults[:100],
        "exact_byte_token_row_tensor_receipt_identity": not faults,
    }


def compare_restart_identity(runs: list[dict[str, Any]]) -> dict[str, Any]:
    identities = []
    for row in runs:
        identities.append(
            {
                "document_count": row.get("document_count"),
                "window_count": row.get("window_count"),
                "materialized_positions": row.get("materialized_positions"),
                "materialization_identity_sha256": row.get(
                    "materialization_identity_sha256"
                ),
                "artifacts": {
                    key: {
                        "bytes": ((row.get("artifacts") or {}).get(key) or {}).get(
                            "bytes"
                        ),
                        "sha256": ((row.get("artifacts") or {}).get(key) or {}).get(
                            "sha256"
                        ),
                    }
                    for key in ARRAY_NAMES
                },
            }
        )
    exact = bool(identities) and all(value == identities[0] for value in identities[1:])
    return {
        "policy": "project_theseus_corpus_materialization_restart_identity_v1",
        "run_count": len(runs),
        "exact_restart": exact,
        "identity_sha256": hashlib.sha256(
            json.dumps(identities[0] if identities else {}, sort_keys=True).encode()
        ).hexdigest(),
        "scratch_outputs_retained": False,
    }


def redact_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "document_count",
        "window_count",
        "materialized_positions",
        "wall_seconds",
        "process_wall_seconds",
        "positions_per_second",
        "peak_rss_bytes",
        "materialization_identity_sha256",
        "artifacts",
    )
    return [{field: row.get(field) for field in fields} for row in runs]


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
