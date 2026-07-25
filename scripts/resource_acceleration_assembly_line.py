#!/usr/bin/env python3
"""Assemble honest end-to-end performance accounting from canonical evidence.

This module belongs to the registered resource-and-acceleration surface.  It does
not benchmark alternate work or invent timings.  Missing station evidence stays
visible so a fast inner loop cannot be promoted as a fast or useful system.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from typing import Any


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


TRAINING_STATIONS = (
    "data_admission",
    "stage_materialization",
    "stage_open_and_batch_index",
    "model_construct",
    "checkpoint_restore",
    "compile_and_warmup",
    "host_batch_preparation",
    "forward_backward_accumulation",
    "final_microbatch_clip_and_optimizer_update",
    "checkpoint_publication",
    "private_development_evaluation",
)

ARCHITECTURE_DECISION_STATIONS = (
    "matched_training",
    "candidate_checkpoint_load",
    "direct_candidate_generation",
    "code_verification",
    "blind_english_scoring",
    "decision_and_receipt_publication",
)

SERVING_STATIONS = (
    "profile_and_policy_resolution",
    "vcm_context_compilation",
    "plan_and_route_selection",
    "resident_model_load",
    "request_queue_and_coalescing",
    "prompt_prefill",
    "autoregressive_decode",
    "deterministic_tool_execution",
    "output_verification",
    "response_publication",
)


def generation_quality_receipt(inference: dict[str, Any]) -> dict[str, Any]:
    """Separate exact route parity from useful generation evidence."""

    cases = inference.get("cases") if isinstance(inference.get("cases"), list) else []
    successful = [
        row
        for row in cases
        if row.get("generation_state") == "GREEN"
        and not row.get("generation_reason")
        and str(row.get("output_sha256") or "") not in {"", EMPTY_SHA256}
    ]
    faulted = [row for row in cases if row.get("generation_state") != "GREEN"]
    arms = sorted({str(row.get("arm_id") or "") for row in successful if row.get("arm_id")})
    case_count = int(inference.get("case_count") or len(cases))
    exact = int(inference.get("exact_parity_case_count") or 0)
    complete_case_receipts = len(cases) == case_count and case_count > 0
    all_useful = complete_case_receipts and len(successful) == case_count
    return {
        "policy": "project_theseus_acceleration_quality_denominator_v1",
        "case_count": case_count,
        "case_receipt_count": len(cases),
        "exact_route_parity_count": exact,
        "successful_nonempty_case_count": len(successful),
        "faulted_case_count": len(faulted),
        "successful_arm_count": len(arms),
        "successful_arms": arms,
        "all_cases_successful_and_nonempty": all_useful,
        "mechanics_parity": complete_case_receipts and exact == case_count,
        "capability_grade_speed_evidence": all_useful and exact == case_count,
        "fault_reasons": count_values(
            str(row.get("generation_reason") or row.get("generation_state") or "unknown")
            for row in faulted
        ),
        "empty_output_sha256": EMPTY_SHA256,
    }


def build_assembly_line(report: dict[str, Any]) -> dict[str, Any]:
    """Build lane-level station ledgers without double-counting unknown work."""

    training = report.get("training") or {}
    corpus = report.get("corpus_to_tensor") or {}
    corpus_performance = corpus.get("performance") or {}
    corpus_sample = corpus.get("sample") or {}
    corpus_parity = corpus.get("parity") or {}
    pair = training.get("paired_canary") or {}
    compiled = pair.get("compiled") or {}
    compiled_runs = compiled.get("runs") if isinstance(compiled.get("runs"), list) else []
    representative_training_run = compiled_runs[0] if compiled_runs else {}
    setup = representative_training_run.get("setup_timings") or {}
    checkpoint = report.get("checkpoint_storage") or {}
    load = report.get("checkpoint_load") or {}
    inference = report.get("inference") or {}
    resident = report.get("resident_runtime") or {}
    refresh = report.get("assistant_context_refresh") or {}
    decision = report.get("architecture_decision_control") or {}
    quality = generation_quality_receipt(inference)

    training_rows = [
        station("data_admission", evidence="No canonical wall-time receipt in acceleration report"),
        station(
            "stage_materialization",
            seconds=number(corpus_performance.get("rust_median_wall_seconds")),
            work_units=int(
                ((corpus_performance.get("rust_runs") or [{}])[0]).get(
                    "materialized_positions"
                )
                or 0
            ),
            bytes_processed=int(corpus_sample.get("input_utf8_bytes") or 0),
            evidence="corpus_to_tensor.performance.rust_median_wall_seconds",
            quality_guards=[
                "exact frozen tokenizer, target offset, row order, windows, labels, masks, and receipts",
                "representative all-category sample; full-stage publication remains separately qualified",
            ],
            quality_ready=bool(
                corpus.get("state") == "GREEN"
                and corpus_parity.get("exact_byte_token_row_tensor_receipt_identity")
            ),
        ),
        station(
            "stage_open_and_batch_index",
            seconds=number(setup.get("total_seconds")),
            evidence="paired_canary.compiled.runs[0].setup_timings.total_seconds",
            quality_guards=["exact metadata, mmap artifact hashes, row ranges, and copy lookup"],
        ),
        station(
            "model_construct",
            seconds=number(representative_training_run.get("model_construct_seconds"))
            or number(load.get("model_construct_seconds")),
            evidence="paired canary training model construction (inference load is secondary)",
            quality_guards=["exact registered model configuration"],
        ),
        station(
            "checkpoint_restore",
            seconds=number(representative_training_run.get("checkpoint_restore_seconds"))
            or number(load.get("weights_load_and_materialize_seconds")),
            bytes_processed=int(load.get("checkpoint_bytes") or 0),
            evidence="paired canary model/optimizer restore and MLX materialization",
            quality_guards=["checkpoint SHA-256 and tensor lineage"],
        ),
        station(
            "compile_and_warmup",
            seconds=derived_compile_seconds(compiled_runs),
            evidence=(
                "Derived as first optimizer-step time minus each run median; Metal trace is still required"
            ),
            measurement_kind="DERIVED" if compiled_runs else "UNMEASURED",
            quality_guards=["same graph, data order, optimizer, and update count"],
        ),
        station(
            "host_batch_preparation",
            seconds=sum_optional(compiled_runs, "host_batch_preparation_seconds_total"),
            evidence="paired_canary.compiled.runs[].host_batch_preparation_seconds_total",
        ),
        station(
            "forward_backward_accumulation",
            seconds=number(compiled.get("compiled_accumulation_seconds_total")),
            work_units=int(compiled.get("warmup_excluded_positions_total") or 0),
            evidence="paired_canary.compiled.compiled_accumulation_seconds_total",
            quality_guards=["bounded full-parameter and loss parity"],
        ),
        station(
            "final_microbatch_clip_and_optimizer_update",
            seconds=number(compiled.get("compiled_update_seconds_total")),
            work_units=int(compiled.get("optimizer_steps_total") or 0),
            evidence="Includes final microbatch forward/backward, clipping, and AdamW update; not optimizer-only",
            quality_guards=["one clip and one update per logical batch"],
        ),
        checkpoint_publication_station(checkpoint),
        station(
            "private_development_evaluation",
            evidence="Loss values exist, but a standalone evaluation wall-time receipt does not",
            quality_guards=["source-disjoint private development split"],
        ),
    ]

    architecture_rows = [
        station(
            "matched_training",
            seconds=number((decision.get("measured_evidence") or {}).get("wall_seconds")),
            evidence="architecture_decision_control.measured_evidence.wall_seconds",
            quality_guards=["matched data, positions, active-forward cost, update cost, and wall time"],
        ),
        station(
            "candidate_checkpoint_load",
            evidence="Review receipts expose per-target load time only after a review is executed",
        ),
        station(
            "direct_candidate_generation",
            seconds=number((decision.get("measured_evidence") or {}).get("generation_seconds")),
            evidence="No completed architecture-review generation receipt",
            quality_guards=["direct model-only; no templates, tools, or fallback returns"],
        ),
        station(
            "code_verification",
            evidence="Verifier duration is not yet aggregated as its own architecture-review station",
            quality_guards=["sandboxed compile, test, render, and hidden-case verification"],
        ),
        station(
            "blind_english_scoring",
            evidence="Rater model load and generation receipts exist only after review execution",
            quality_guards=["local-only, architecture-blind, reference-blind scoring"],
        ),
        station(
            "decision_and_receipt_publication",
            evidence="No completed review decision receipt",
            quality_guards=["immutable lineage and weak-tail reporting"],
        ),
    ]

    cold_commands = {
        str(row.get("id")): float(row.get("runtime_ms") or 0.0) / 1000.0
        for row in (refresh.get("cold_commands") or [])
    }
    serving_rows = [
        station(
            "profile_and_policy_resolution",
            evidence="Combined with assistant refresh orchestration; exclusive timing is unmeasured",
        ),
        station(
            "vcm_context_compilation",
            seconds=cold_commands.get("vcm_context_governor", 0.0)
            + cold_commands.get("vcm_task_context_bridge", 0.0),
            warm_seconds=warm_command_seconds(refresh, ("vcm_context_governor", "vcm_task_context_bridge")),
            evidence="assistant_context_refresh command receipts",
            quality_guards=["content, policy, capability, snapshot, and freshness identity"],
        ),
        station(
            "plan_and_route_selection",
            seconds=cold_commands.get("plan_compiler"),
            warm_seconds=warm_command_seconds(refresh, ("plan_compiler",)),
            evidence="assistant_context_refresh plan_compiler receipt",
        ),
        station(
            "resident_model_load",
            seconds=number((resident.get("runtime") or {}).get("load_seconds"))
            or number(load.get("weights_load_and_materialize_seconds")),
            bytes_processed=int(load.get("checkpoint_bytes") or 0),
            evidence="resident_runtime.runtime.load_seconds",
            quality_guards=["one exact checkpoint load per resident process"],
        ),
        station(
            "request_queue_and_coalescing",
            seconds=number(((resident.get("continuous_batching") or {}).get("concurrent_coalesced_seconds"))),
            work_units=int(((resident.get("continuous_batching") or {}).get("request_count")) or 0),
            evidence="resident_runtime.continuous_batching.concurrent_coalesced_seconds",
            quality_guards=["exact serial/batched state, reason, output, and token parity"],
        ),
        station(
            "prompt_prefill",
            seconds=number(resident.get("uncached_prefill_seconds")),
            warm_seconds=number(resident.get("cached_prefill_seconds")),
            evidence="resident_runtime uncached/cached prefill receipts",
            quality_guards=["cache speedups excluded from novel-request decode headline"],
        ),
        station(
            "autoregressive_decode",
            seconds=number((inference.get("optimized_latency_seconds") or {}).get("total")),
            p50_seconds=number((inference.get("optimized_latency_seconds") or {}).get("p50")),
            p95_seconds=number((inference.get("optimized_latency_seconds") or {}).get("p95")),
            work_units=int(inference.get("case_count") or 0),
            evidence="inference.optimized_latency_seconds",
            quality_guards=[
                "exact route parity",
                "capability-grade speed requires every measured output to be successful and nonempty",
            ],
            quality_ready=bool(quality["capability_grade_speed_evidence"]),
        ),
        station(
            "deterministic_tool_execution",
            seconds=cold_commands.get("deterministic_tool_registry"),
            warm_seconds=warm_command_seconds(refresh, ("deterministic_tool_registry",)),
            evidence="Registry refresh only; task-specific tool execution remains unmeasured",
            measurement_kind="PARTIAL" if "deterministic_tool_registry" in cold_commands else "UNMEASURED",
        ),
        station(
            "output_verification",
            seconds=cold_commands.get("private_verifier_spine_smoke"),
            warm_seconds=warm_command_seconds(refresh, ("private_verifier_spine_smoke",)),
            evidence="Verifier refresh smoke only; task-specific output verification remains unmeasured",
            measurement_kind="PARTIAL" if "private_verifier_spine_smoke" in cold_commands else "UNMEASURED",
            quality_guards=["verification may not be skipped for speed"],
        ),
        station("response_publication", evidence="No canonical exclusive timing receipt"),
    ]

    lanes = {
        "training_feedback": lane(TRAINING_STATIONS, training_rows),
        "architecture_decision": lane(ARCHITECTURE_DECISION_STATIONS, architecture_rows),
        "interactive_serving": lane(SERVING_STATIONS, serving_rows),
    }
    ranked = sorted(
        (
            {
                "lane": lane_id,
                "station_id": row["station_id"],
                "observed_seconds": row["timing"]["cold_seconds"],
                "measurement_kind": row["measurement_kind"],
                "quality_ready": row["quality_ready"],
            }
            for lane_id, payload in lanes.items()
            for row in payload["stations"]
            if row["timing"]["cold_seconds"] is not None
        ),
        key=lambda row: float(row["observed_seconds"] or 0.0),
        reverse=True,
    )
    return {
        "policy": "project_theseus_capability_critical_assembly_line_v1",
        "primary_metric": "wall_time_to_verified_useful_output_or_defensible_architecture_decision",
        "secondary_metrics": [
            "accepted_verified_outputs_per_second",
            "same_semantics_optimizer_positions_per_second",
            "interactive_p50_p95_seconds",
            "peak_unified_memory_bytes",
            "bytes_read_and_written",
        ],
        "lanes": lanes,
        "generation_quality_denominator": quality,
        "ranked_observed_stations": ranked,
        "measurement_complete": all(payload["measurement_complete"] for payload in lanes.values()),
        "quality_complete": all(payload["quality_complete"] for payload in lanes.values()),
        "promotion_rule": (
            "No station or lane is performance-qualified by faulted, empty, skipped, reduced-budget, "
            "or semantically changed work. Mechanics parity remains useful evidence but is not capability."
        ),
        "next_measurements": next_measurements(lanes),
        "optimization_backlog": optimization_backlog(),
        "candidate_disposition_ledger": candidate_disposition_ledger(report),
        "platform_dispositions": [
            {
                "platform": "macOS Apple Silicon",
                "state": "EMPIRICALLY_QUALIFIED_CURRENT_HOST",
                "routes": ["Rust exact/KERC preprocessing", "MLX FP32 compiled training"],
            },
            {
                "platform": "macOS Intel",
                "state": "DEFERRED_HARDWARE",
                "routes": ["Rust/CPU preprocessing expected portable", "MLX inapplicable"],
            },
            {
                "platform": "Windows CUDA/CPU",
                "state": "DEFERRED_UNREACHABLE_HARDWARE",
                "routes": ["independent arm/evaluation fanout; no parity claim"],
            },
            {
                "platform": "Linux CUDA/CPU",
                "state": "DEFERRED_UNAVAILABLE_HARDWARE",
                "routes": ["independent arm/evaluation fanout; no parity claim"],
            },
        ],
    }


def candidate_disposition_ledger(report: dict[str, Any]) -> dict[str, Any]:
    """Give the finite T0P candidate set an explicit, evidence-bound outcome."""

    adoption = report.get("adoption") or {}
    training = report.get("training") or {}
    corpus = report.get("corpus_to_tensor") or {}
    ingestion = corpus.get("ingestion_parallelism_and_scanner") or {}
    scanner = ingestion.get("scanner") or {}
    compressed = ingestion.get("compressed_input") or {}
    parallel = ingestion.get("bounded_parallel_materialization") or {}
    content_cache = ingestion.get("content_addressed_cache") or {}
    resident = report.get("resident_runtime") or {}
    checkpoint = report.get("checkpoint_storage") or {}
    inference_quality = generation_quality_receipt(report.get("inference") or {})
    bf16_resume = training.get("bf16_checkpoint_resume") or {}
    fp32_resume = training.get("fp32_checkpoint_resume") or {}
    rows = [
        disposition(
            "exact_corpus_to_tensor_rust",
            "ADOPTED" if adoption.get("rust_exact_corpus_to_tensor") == "QUALIFIED_FROZEN_LINEAGE" else "DEFERRED",
            "corpus_to_tensor",
            "exact frozen-lineage tensors, receipts, restart identity, and corruption rejection",
        ),
        disposition(
            "typed_kerc_dual_space_rust",
            "ADOPTED" if adoption.get("rust_kerc_dual_space_encoding") == "QUALIFIED_TYPED_PREPROCESSING" else "DEFERRED",
            "corpus_to_tensor.kerc_dual_space",
            "typed V_K/V_P preprocessing only; no KERC capability credit",
        ),
        disposition(
            "simd_swar_exact_scanner",
            str(scanner.get("disposition") or "DEFERRED"),
            "corpus_to_tensor.ingestion_parallelism_and_scanner.scanner",
            "AArch64 NEON runs receive speed credit only with scalar differential identity",
        ),
        disposition(
            "compressed_jsonl_direct_ingestion",
            "ADOPTED" if compressed.get("exact_artifact_parity") is True else "DEFERRED",
            "corpus_to_tensor.ingestion_parallelism_and_scanner.compressed_input",
            "plain, gzip, and zstd streams must materialize byte-identical tensors",
        ),
        disposition(
            "parquet_source_ingestion",
            "INAPPLICABLE_CURRENT_LINEAGE",
            "canonical_pretrain_index_v2",
            "the frozen source index resolves JSONL/raw-text records, not Parquet; implement only with an admitted Parquet source",
        ),
        disposition(
            "bounded_parallel_corpus_encoding",
            str(parallel.get("disposition") or "DEFERRED"),
            "corpus_to_tensor.ingestion_parallelism_and_scanner.bounded_parallel_materialization",
            "fixed-size chunks reconstruct source order exactly; adoption also requires a material representative gain",
        ),
        disposition(
            "persistent_hostwide_encoded_shard_cache",
            "ADOPTED" if content_cache.get("state") == "GREEN" else "DEFERRED",
            "corpus_to_tensor.ingestion_parallelism_and_scanner.content_addressed_cache",
            "cache hits verify immutable content identity and artifact hashes under an explicit host budget; corruption fails closed",
        ),
        disposition(
            "successor_bpe_tokenizer",
            "INAPPLICABLE_CURRENT_LINEAGE",
            "frozen_abi",
            "would change token IDs and checkpoint meaning; belongs to a prospectively frozen successor",
        ),
        disposition(
            "compiled_fp32_microbatch4",
            "ADOPTED" if fp32_resume.get("state") == "GREEN" else "DEFERRED",
            "training.fp32_checkpoint_resume",
            "resume mechanics qualified; sustained 2x target remains a separate requirement",
        ),
        disposition(
            "compiled_fp32_microbatch8",
            "REJECTED",
            "roadmap:T0P.3",
            "full-state parity held but repeated speed stayed below the 2x target",
        ),
        disposition(
            "bf16_compute_fp32_master",
            "REJECTED" if bf16_resume.get("state") == "RED" else "DEFERRED",
            "training.bf16_checkpoint_resume",
            "same-start BF16 updates diverged before serialization; short speed does not override reproducibility",
        ),
        disposition(
            "deferred_microbatch_graph_sync",
            "REJECTED",
            "roadmap:T0P.4",
            "lower speed and higher unified memory than the retained synchronization boundary",
        ),
        disposition(
            "redundant_post_update_state_sync_removal",
            "ADOPTED",
            "runtime/tmp/resource_acceleration_redundant_sync_pair.json",
            "the compiled optimizer step is already synchronized; removal preserves loss and full-state parity, while the canonical full qualification owns the sustained speed claim",
        ),
        disposition(
            "split_accumulation_update_graph",
            "REJECTED",
            "runtime/tmp/resource_acceleration_split_graph_canary.json",
            "the paired ratio was inflated by a slower eager control and absolute compiled throughput regressed below the retained single-graph route",
        ),
        disposition(
            "canonical_full_metal_trace",
            "DEFERRED",
            "metal_trace",
            "one post-compile optimizer step still emitted 38.6 GB; use a bounded profiler before kernel work",
        ),
        disposition(
            "mlx_fast_rope_training_only",
            "ADOPTED",
            "training.paired_canary",
            "bounded update parity and repeated training-only benefit; serving remains manual reference",
        ),
        disposition(
            "mlx_fast_rope_serving",
            "REJECTED",
            "roadmap:T0P.4",
            "changed serving token paths",
        ),
        disposition(
            "fused_qkv_swiglu_projection",
            "REJECTED",
            "roadmap:T0P.4",
            "measured below the retained route despite parameter-preserving implementation",
        ),
        disposition(
            "safetensors_checkpoint",
            "ADOPTED" if checkpoint.get("exact_tensor_parity") is True else "DEFERRED",
            "checkpoint_storage",
            "exact tensor manifest and materially faster load; publication overlap remains separate",
        ),
        disposition(
            "asynchronous_checkpoint_publication",
            "DEFERRED",
            "assembly_line.training_feedback.checkpoint_publication",
            "requires immutable snapshot, peak-memory, crash-boundary, and recovery evidence",
        ),
        disposition(
            "device_filtered_batched_beam_decode",
            "ADOPTED" if inference_quality["capability_grade_speed_evidence"] and adoption.get("batched_beam_device_filter_preprune") == "QUALIFIED" else "DEFERRED",
            "inference",
            "mechanics parity alone is insufficient when measured outputs are empty or faulted",
        ),
        disposition(
            "resident_prefix_completion_cache",
            "ADOPTED_EVALUATION_RUNTIME" if resident.get("trigger_state") == "GREEN" else "DEFERRED",
            "resident_runtime",
            "production serving and utility remain unclaimed",
        ),
        disposition(
            "sequence_axis_kv_preallocation",
            "REJECTED",
            "roadmap:T0P.7",
            "1.009x bounded stress result was immaterial",
        ),
        disposition(
            "multi_node_synchronous_dense_training",
            "INAPPLICABLE_CURRENT_HARDWARE",
            "hardware",
            "single-node environment; independent arm/eval fanout is preferred when peers are available",
        ),
    ]
    counts = count_values(row["disposition"] for row in rows)
    return {
        "policy": "project_theseus_finite_performance_candidate_ledger_v1",
        "candidate_count": len(rows),
        "disposition_counts": counts,
        "all_candidates_dispositioned": all(row["disposition"] for row in rows),
        "rows": rows,
        "claim_boundary": "a disposition closes one implementation candidate, not the station or capability claim",
    }


def disposition(
    candidate_id: str, outcome: str, evidence: str, boundary: str
) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "disposition": outcome,
        "evidence": evidence,
        "boundary": boundary,
    }


def station(
    station_id: str,
    *,
    seconds: float | None = None,
    warm_seconds: float | None = None,
    p50_seconds: float | None = None,
    p95_seconds: float | None = None,
    work_units: int = 0,
    bytes_processed: int = 0,
    evidence: str,
    measurement_kind: str | None = None,
    quality_guards: list[str] | None = None,
    quality_ready: bool = True,
) -> dict[str, Any]:
    if measurement_kind is None:
        measurement_kind = "MEASURED" if seconds is not None else "UNMEASURED"
    throughput = (
        round(work_units / seconds, 6)
        if seconds is not None and seconds > 0.0 and work_units > 0
        else None
    )
    return {
        "station_id": station_id,
        "measurement_kind": measurement_kind,
        "timing": {
            "cold_seconds": rounded(seconds),
            "warm_seconds": rounded(warm_seconds),
            "p50_seconds": rounded(p50_seconds),
            "p95_seconds": rounded(p95_seconds),
        },
        "work_units": int(work_units),
        "throughput_units_per_second": throughput,
        "bytes_processed": int(bytes_processed),
        "quality_ready": bool(quality_ready),
        "quality_guards": list(quality_guards or []),
        "evidence": evidence,
    }


def lane(expected: tuple[str, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["station_id"]: row for row in rows}
    if tuple(by_id) != expected:
        raise ValueError("assembly-line station order or membership changed")
    unmeasured = [row["station_id"] for row in rows if row["measurement_kind"] == "UNMEASURED"]
    partial = [row["station_id"] for row in rows if row["measurement_kind"] in {"PARTIAL", "DERIVED"}]
    quality_blocked = [row["station_id"] for row in rows if not row["quality_ready"]]
    observed = sum(
        float(row["timing"]["cold_seconds"] or 0.0)
        for row in rows
        if row["timing"]["cold_seconds"] is not None
    )
    return {
        "stations": rows,
        "observed_seconds_not_end_to_end": round(observed, 6),
        "unmeasured_stations": unmeasured,
        "partial_or_derived_stations": partial,
        "quality_blocked_stations": quality_blocked,
        "measurement_complete": not unmeasured and not partial,
        "quality_complete": not quality_blocked,
        "critical_path_claimed": False,
    }


def checkpoint_publication_station(checkpoint: dict[str, Any]) -> dict[str, Any]:
    formats = checkpoint.get("formats") or {}
    selected = formats.get("safetensors") or {}
    seconds = number(selected.get("serialization_seconds"))
    hash_seconds = number(selected.get("content_hash_seconds"))
    if seconds is not None and hash_seconds is not None:
        seconds += hash_seconds
    return station(
        "checkpoint_publication",
        seconds=seconds,
        bytes_processed=int(selected.get("bytes") or 0),
        evidence="safetensors serialization plus content hashing; optimizer-state publication remains unmeasured",
        measurement_kind="PARTIAL" if seconds is not None else "UNMEASURED",
        quality_guards=["atomic replace, exact tensor manifest, restart and resume integrity"],
    )


def derived_compile_seconds(rows: list[dict[str, Any]]) -> float | None:
    values = []
    for row in rows:
        first = number(row.get("first_optimizer_step_seconds"))
        median = number(row.get("median_optimizer_step_seconds"))
        if first is not None and median is not None:
            values.append(max(0.0, first - median))
    return statistics.mean(values) if values else None


def sum_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [number(row.get(key)) for row in rows]
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def warm_command_seconds(refresh: dict[str, Any], ids: tuple[str, ...]) -> float | None:
    rows = refresh.get("warm_commands") if isinstance(refresh.get("warm_commands"), list) else []
    selected = [float(row.get("runtime_ms") or 0.0) / 1000.0 for row in rows if row.get("id") in ids]
    return sum(selected) if selected else None


def next_measurements(lanes: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for lane_id, payload in lanes.items():
        for station_id in payload["unmeasured_stations"]:
            rows.append({"lane": lane_id, "station_id": station_id, "reason": "no exclusive canonical timing"})
        for station_id in payload["partial_or_derived_stations"]:
            rows.append({"lane": lane_id, "station_id": station_id, "reason": "exclusive measurement incomplete"})
    return rows


def optimization_backlog() -> list[dict[str, Any]]:
    return [
        {
            "priority": 1,
            "stations": ["compile_and_warmup", "forward_backward_accumulation", "final_microbatch_clip_and_optimizer_update"],
            "candidate": "Capture an MLX Metal trace and exported graph for the exact compiled step; optimize only the measured dominant kernels or synchronization barriers.",
            "semantics_risk": "LOW_IF_GRAPH_IDENTICAL",
        },
        {
            "priority": 2,
            "stations": ["code_verification", "blind_english_scoring"],
            "candidate": "Run independent sandbox verifiers concurrently with bounded per-tool limits and score all opaque candidate packets in one rater-model load pass.",
            "semantics_risk": "LOW_WITH_ORDER_INDEPENDENCE_AND_IDENTICAL_OUTPUTS",
        },
        {
            "priority": 3,
            "stations": ["candidate_checkpoint_load", "direct_candidate_generation"],
            "candidate": "Keep review models resident and batch prompts by model, prefix, and length while preserving one direct candidate per case.",
            "semantics_risk": "LOW_WITH_EXACT_TOKEN_PATH_PARITY",
        },
        {
            "priority": 4,
            "stations": ["checkpoint_publication"],
            "candidate": "Measure time-based cadence and immutable snapshot publication; consider asynchronous flush only after peak unified-memory and crash-recovery proof.",
            "semantics_risk": "MEDIUM_RECOVERY_CRITICAL",
        },
        {
            "priority": 5,
            "stations": ["prompt_prefill", "autoregressive_decode"],
            "candidate": "Qualify prefix-homogeneous batching, paged/rotating KV storage, exact speculative decoding, and MTP separately on successful outputs.",
            "semantics_risk": "MEDIUM_REQUIRES_QUALITY_AND_DISTRIBUTION_PARITY",
        },
        {
            "priority": 6,
            "stations": ["forward_backward_accumulation", "final_microbatch_clip_and_optimizer_update"],
            "candidate": "Evaluate memory-efficient optimizer state or alternative precision only as a preregistered learning-equivalence experiment, never as a silent speed patch.",
            "semantics_risk": "HIGH_CHANGES_OPTIMIZATION_DYNAMICS",
        },
    ]


def count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rounded(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
