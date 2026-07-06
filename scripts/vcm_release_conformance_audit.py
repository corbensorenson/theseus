"""Audit Theseus VCM artifacts against the Virtual Context Memory v1.0 packet.

The audit is intentionally conservative. It checks the local implementation
against packet profiles and invariants without claiming native runtime/KV-cache
conformance before those mechanisms exist.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_PACKET_SOURCE = ROOT / "docs" / "reference" / "virtual_context_memory_v1.0" / "source" / "Virtual_Context_Memory_v1.0.md"
DEFAULT_PACKET_AUDIT = ROOT / "docs" / "reference" / "virtual_context_memory_v1.0" / "release" / "Virtual_Context_Memory_v1.0_Release_Audit.md"
DEFAULT_OUT = REPORTS / "vcm_release_conformance_audit.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "vcm_release_conformance_audit.md"


INVARIANT_TEXT = {
    1: "Every model-visible durable memory span resolves to an immutable page version and provenance role.",
    2: "Every derived page representation exposes a certificate, authority ceiling, declared loss, and fallback path.",
    3: "Every task uses a named snapshot and receives read-your-writes behavior.",
    4: "Every conflict is represented, adjudicated, or explicitly excluded with a reason.",
    5: "Every prefetch enters non-model-visible staging before promotion.",
    6: "Every promotion passes capability, purpose, taint, freshness, use-contract, and authority gates.",
    7: "Every tool action receives current authorization independent of memory text.",
    8: "Every runtime cache is keyed by complete source, representation, model, policy, principal, redaction, permission, and snapshot keys.",
    9: "Every page deletion initiates graph-based descendant and cache handling.",
    10: "Every user preference records scope, evidence mode, confidence, and correction controls.",
    11: "Every context switch checkpoints dirty state and mounts a versioned root.",
    12: "Every page-fault denial or failure produces a safe fallback rather than unsupported reconstruction.",
    13: "No transformation or cache materialization increases behavioral authority beyond source and policy ceilings.",
    14: "Protected minimum overflow returns explicit unsafe-fit rather than dropping a mandatory page.",
    15: "Semantic materialization from a lossy derivative creates a new certified representation; it is never exact decompression.",
    16: "Learned scoring, compression, and prefetch policies operate inside deterministic constraints.",
    17: "Low retrieval frequency alone cannot trigger permanent deletion.",
    18: "A page cannot self-designate as mandatory, pinned, trusted, or privileged.",
    19: "Reusable prefix or KV objects are valid only under complete ordered runtime keys.",
    20: "Online retention, compression, and prefetch decisions replay from decision-time observable features.",
    21: "Runtime future-reuse promises require accepted resident-materialization claims and fail-closed outcomes.",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-source", default=str(DEFAULT_PACKET_SOURCE))
    parser.add_argument("--packet-audit", default=str(DEFAULT_PACKET_AUDIT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT))
    args = parser.parse_args()

    report = build_report(packet_source=Path(args.packet_source), packet_audit=Path(args.packet_audit))
    write_json(Path(args.out), report)
    write_text(Path(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["core_profiles_ready"] else 2


def build_report(*, packet_source: Path, packet_audit: Path) -> dict[str, Any]:
    packet_text = read_text(packet_source)
    audit_text = read_text(packet_audit)
    compiled = read_json(REPORTS / "virtual_context_compiled_context.json")
    probe = read_json(REPORTS / "virtual_context_memory_probe.json")
    bench = read_json(REPORTS / "virtual_context_memory_bench.json")
    training = read_json(REPORTS / "virtual_context_memory_training_admission.json")
    consumers = read_json(REPORTS / "virtual_context_memory_consumer_audit.json")
    recovery = read_json(REPORTS / "vcm_context_recovery_benchmark.json")
    graph = read_json(REPORTS / "virtual_context_memory_graph.json")
    snapshots = read_json(REPORTS / "virtual_context_memory_snapshots.json")
    pages = read_jsonl(REPORTS / "virtual_context_memory_pages.jsonl")
    prefetch_regret = read_json(REPORTS / "vcm_prefetch_regret_audit.json")
    runtime_readiness = read_json(REPORTS / "vcm_runtime_claim_readiness.json")
    runtime_lifecycle = read_json(REPORTS / "vcm_runtime_cache_lifecycle.json")
    native_runtime_probe = read_json(REPORTS / "vcm_native_runtime_probe.json")
    task_context_bridge = read_json(REPORTS / "vcm_task_context_bridge.json")
    public_memory_calibration = read_json(REPORTS / "vcm_public_memory_calibration.json")
    public_memory_readiness = read_json(REPORTS / "vcm_public_memory_readiness_audit.json")
    public_memory_prompt_calibration = read_json(REPORTS / "vcm_public_memory_prompt_calibration.json")
    public_memory_private_repair = read_json(REPORTS / "vcm_public_memory_private_residual_repair.json")
    private_lme_residual = read_json(REPORTS / "vcm_longmemeval_private_residual_curriculum.json")
    vcm_evidence_gauntlet = read_json(REPORTS / "vcm_evidence_gauntlet.json")
    hard_memory_private = read_json(REPORTS / "vcm_hard_memory_private_analogues.json")
    hard_memory_readiness = read_json(REPORTS / "vcm_hard_memory_benchmark_readiness.json")

    checks = derive_checks(
        packet_text=packet_text,
        audit_text=audit_text,
        compiled=compiled,
        probe=probe,
        bench=bench,
        training=training,
        consumers=consumers,
        recovery=recovery,
        graph=graph,
        snapshots=snapshots,
        pages=pages,
        prefetch_regret=prefetch_regret,
        runtime_readiness=runtime_readiness,
        runtime_lifecycle=runtime_lifecycle,
        native_runtime_probe=native_runtime_probe,
        task_context_bridge=task_context_bridge,
        public_memory_calibration=public_memory_calibration,
        public_memory_readiness=public_memory_readiness,
        public_memory_prompt_calibration=public_memory_prompt_calibration,
        public_memory_private_repair=public_memory_private_repair,
        private_lme_residual=private_lme_residual,
        vcm_evidence_gauntlet=vcm_evidence_gauntlet,
    )
    profiles = build_profiles(checks)
    invariants = build_invariants(checks)
    required_profiles = ["VCM-Core", "VCM-Governed", "VCM-Transactional"]
    core_profiles_ready = all(profiles[name]["status"] == "GREEN" for name in required_profiles)
    hard_failures = [
        {"kind": "profile", "id": name, "evidence": profiles[name]["evidence"]}
        for name in required_profiles
        if profiles[name]["status"] != "GREEN"
    ]
    hard_failures.extend(
        {"kind": "invariant", "id": row["id"], "evidence": row["evidence"]}
        for row in invariants
        if row["required_for_core"] and row["status"] != "GREEN"
    )
    native_runtime_claimable = bool(checks.get("native_runtime_claimable"))
    summary = {
        "packet_present": bool(packet_text),
        "packet_declares_conceptual_status": bool(checks.get("packet_conceptual")),
        "profile_states": {name: row["status"] for name, row in profiles.items()},
        "core_profiles_ready": core_profiles_ready and not hard_failures,
        "hard_failure_count": len(hard_failures),
        "runtime_profile_claimed": native_runtime_claimable,
        "predictive_profile_state": profiles["VCM-Predictive"]["status"],
        "external_inference_calls": total_external_inference(compiled, probe, bench, training, recovery),
        "public_training_rows_written": int(probe_summary(probe).get("public_training_rows_written") or 0),
        "public_calibration_runs": int(probe_summary(probe).get("public_calibration_runs") or 0),
        "fallback_return_count": int(probe_summary(probe).get("fallback_return_count") or 0),
        "prefetch_regret_state": prefetch_regret.get("trigger_state"),
        "prefetch_regret": get_path(prefetch_regret, ["summary", "prefetch_regret"]),
        "runtime_readiness_state": runtime_readiness.get("trigger_state"),
        "runtime_cache_key_complete_rate": get_path(runtime_readiness, ["summary", "cache_key_complete_rate"]),
        "runtime_cache_lifecycle_state": runtime_lifecycle.get("trigger_state"),
        "runtime_cache_reuse_hit_rate": get_path(runtime_lifecycle, ["summary", "reuse_hit_rate"]),
        "runtime_cache_snapshot_invalidation_miss_rate": get_path(runtime_lifecycle, ["summary", "snapshot_invalidation_miss_rate"]),
        "runtime_cache_policy_invalidation_miss_rate": get_path(runtime_lifecycle, ["summary", "policy_invalidation_miss_rate"]),
        "runtime_cache_lifecycle_records": get_path(runtime_lifecycle, ["summary", "cache_records"]),
        "native_runtime_probe_state": native_runtime_probe.get("trigger_state"),
        "native_runtime_claimable": get_path(native_runtime_probe, ["summary", "native_runtime_claimable"]),
        "native_prefix_kv_lifecycle_test_passed": get_path(native_runtime_probe, ["summary", "native_prefix_kv_lifecycle_test_passed"]),
        "native_runtime_route_metadata_ready": get_path(native_runtime_probe, ["summary", "hardware_aware_runtime_route_metadata_ready"]),
        "native_runtime_recommended_backend": get_path(native_runtime_probe, ["summary", "recommended_backend"]),
        "native_runtime_recommended_execution_backend": get_path(native_runtime_probe, ["summary", "recommended_execution_backend"]),
        "native_runtime_recommended_python": get_path(native_runtime_probe, ["summary", "recommended_python"]),
        "native_runtime_claim_scope": get_path(native_runtime_probe, ["summary", "native_runtime_claim_scope"]),
        "native_runtime_claim_backend": get_path(native_runtime_probe, ["summary", "native_runtime_claim_backend"]),
        "native_runtime_claim_device": get_path(native_runtime_probe, ["summary", "native_runtime_claim_device"]),
        "native_runtime_claim_backend_matches_recommended_execution_backend": get_path(
            native_runtime_probe, ["summary", "native_runtime_claim_backend_matches_recommended_execution_backend"]
        ),
        "mlx_tensor_descriptor_lifecycle_test_passed": get_path(
            native_runtime_probe, ["summary", "mlx_tensor_descriptor_lifecycle_test_passed"]
        ),
        "mlx_tensor_descriptor_runtime_kind": get_path(
            native_runtime_probe, ["summary", "mlx_tensor_descriptor_runtime_kind"]
        ),
        "mlx_tensor_descriptor_device": get_path(native_runtime_probe, ["summary", "mlx_tensor_descriptor_device"]),
        "recommended_backend_runtime_descriptor_lifecycle_claimable": get_path(
            native_runtime_probe, ["summary", "recommended_backend_runtime_descriptor_lifecycle_claimable"]
        ),
        "scheduler_vcm_descriptor_route_allowed_for_recommended_backend": get_path(
            native_runtime_probe, ["summary", "scheduler_vcm_descriptor_route_allowed_for_recommended_backend"]
        ),
        "recommended_backend_native_runtime_claimable": get_path(
            native_runtime_probe, ["summary", "recommended_backend_native_runtime_claimable"]
        ),
        "scheduler_native_kv_route_allowed_for_recommended_backend": get_path(
            native_runtime_probe, ["summary", "scheduler_native_kv_route_allowed_for_recommended_backend"]
        ),
        "scheduler_native_kv_route_fail_closed": get_path(
            native_runtime_probe, ["summary", "scheduler_native_kv_route_fail_closed"]
        ),
        "accelerator_kv_parity_claimed": get_path(native_runtime_probe, ["summary", "accelerator_kv_parity_claimed"]),
        "mlx_native_kv_parity_claimed": get_path(native_runtime_probe, ["summary", "mlx_native_kv_parity_claimed"]),
        "cuda_native_kv_parity_claimed": get_path(native_runtime_probe, ["summary", "cuda_native_kv_parity_claimed"]),
        "metal_native_kv_parity_claimed": get_path(native_runtime_probe, ["summary", "metal_native_kv_parity_claimed"]),
        "native_runtime_blocker_count": get_path(native_runtime_probe, ["summary", "blocker_count"]),
        "task_context_bridge_state": task_context_bridge.get("trigger_state"),
        "task_context_family_count": get_path(task_context_bridge, ["summary", "task_family_count"]),
        "task_context_ready_family_count": get_path(task_context_bridge, ["summary", "ready_task_family_count"]),
        "task_context_high_priority_ready": get_path(task_context_bridge, ["summary", "high_priority_ready_count"]),
        "task_context_high_priority_total": get_path(task_context_bridge, ["summary", "high_priority_task_family_count"]),
        "task_context_bridge_clean": bool(checks.get("task_context_bridge_clean")),
        "public_memory_calibration_state": public_memory_calibration.get("trigger_state"),
        "public_memory_calibration_mode": public_memory_calibration.get("calibration_mode"),
        "public_memory_readiness_state": public_memory_readiness.get("trigger_state"),
        "public_memory_readiness_allowed": public_memory_readiness.get("public_calibration_allowed"),
        "public_memory_readiness_clean": bool(checks.get("public_memory_readiness_clean")),
        "public_memory_prompt_calibration_state": public_memory_prompt_calibration.get("trigger_state"),
        "public_memory_prompt_vcm_on_pass_rate": get_path(public_memory_prompt_calibration, ["summary", "vcm_on_pass_rate"]),
        "public_memory_prompt_vcm_off_pass_rate": get_path(public_memory_prompt_calibration, ["summary", "vcm_off_pass_rate"]),
        "public_memory_prompt_vcm_over_flat_tail_delta": get_path(public_memory_prompt_calibration, ["summary", "vcm_over_flat_tail_delta"]),
        "public_memory_prompt_vcm_over_best_non_vcm_delta": get_path(public_memory_prompt_calibration, ["summary", "vcm_over_best_non_vcm_delta"]),
        "public_memory_prompt_best_non_vcm": get_path(public_memory_prompt_calibration, ["summary", "best_non_vcm_memory_system", "system"]),
        "public_memory_prompt_off_only_wins": get_path(public_memory_prompt_calibration, ["summary", "win_counts", "vcm_off"]),
        "public_memory_prompt_item_manifest": get_path(public_memory_prompt_calibration, ["quarantine", "item_manifest"]),
        "public_memory_prompt_item_manifest_hash": get_path(public_memory_prompt_calibration, ["quarantine", "item_manifest_hash"]),
        "public_memory_prompt_source_context_token_distribution": get_path(public_memory_prompt_calibration, ["summary", "source_context_token_distribution"]),
        "public_memory_prompt_per_length_bucket": get_path(public_memory_prompt_calibration, ["summary", "per_length_bucket"]),
        "public_memory_prompt_forbidden_overlap_counts": get_path(public_memory_prompt_calibration, ["summary", "forbidden_overlap_counts"]),
        "public_memory_prompt_no_regression": bool(checks.get("public_memory_prompt_no_regression")),
        "public_memory_prompt_private_repair_state": public_memory_private_repair.get("trigger_state"),
        "private_longmemeval_residual_curriculum_state": private_lme_residual.get("trigger_state"),
        "private_longmemeval_residual_vcm_pass_rate": get_path(private_lme_residual, ["summary", "vcm_on_pass_rate"]),
        "private_longmemeval_residual_min_type_pass_rate": get_path(private_lme_residual, ["summary", "minimum_major_question_type_pass_rate"]),
        "private_longmemeval_residual_vcm_over_best_non_vcm_delta": get_path(private_lme_residual, ["summary", "vcm_over_best_single_non_vcm_delta"]),
        "vcm_evidence_gauntlet_state": vcm_evidence_gauntlet.get("trigger_state"),
        "vcm_evidence_gauntlet_cases": get_path(vcm_evidence_gauntlet, ["summary", "case_count"]),
        "vcm_evidence_gauntlet_vcm_pass_rate": get_path(vcm_evidence_gauntlet, ["summary", "vcm_on_pass_rate"]),
        "vcm_evidence_gauntlet_best_single_non_vcm": get_path(vcm_evidence_gauntlet, ["summary", "best_single_non_vcm_pass_rate"]),
        "vcm_evidence_gauntlet_delta": get_path(vcm_evidence_gauntlet, ["summary", "vcm_over_best_single_non_vcm_delta"]),
        "vcm_evidence_gauntlet_min_family": get_path(vcm_evidence_gauntlet, ["summary", "minimum_major_family_pass_rate"]),
        "vcm_evidence_gauntlet_abstention": get_path(vcm_evidence_gauntlet, ["summary", "abstention"]),
        "hard_memory_private_state": hard_memory_private.get("trigger_state"),
        "hard_memory_private_cases": get_path(hard_memory_private, ["summary", "case_count"]),
        "hard_memory_private_family_count": get_path(hard_memory_private, ["summary", "family_count"]),
        "hard_memory_private_length_bucket_count": get_path(hard_memory_private, ["summary", "length_bucket_count"]),
        "hard_memory_private_vcm_pass_rate": get_path(hard_memory_private, ["summary", "vcm_on_pass_rate"]),
        "hard_memory_private_best_single_non_vcm": get_path(hard_memory_private, ["summary", "best_single_non_vcm_pass_rate"]),
        "hard_memory_private_delta": get_path(hard_memory_private, ["summary", "vcm_over_best_single_non_vcm_delta"]),
        "hard_memory_private_min_family": get_path(hard_memory_private, ["summary", "minimum_family_pass_rate"]),
        "hard_memory_readiness_state": hard_memory_readiness.get("trigger_state"),
        "hard_memory_readiness_public_rows": get_path(hard_memory_readiness, ["summary", "current_public_prompt_rows_scored"]),
        "hard_memory_readiness_public_target": get_path(hard_memory_readiness, ["summary", "public_row_target"]),
        "hard_memory_readiness_metadata_ready_count": get_path(hard_memory_readiness, ["summary", "metadata_ready_count"]),
        "hard_memory_readiness_blocked_or_queued_count": get_path(hard_memory_readiness, ["summary", "blocked_or_queued_count"]),
    }
    if not summary["core_profiles_ready"]:
        trigger_state = "RED"
    elif profiles["VCM-Predictive"]["status"] != "GREEN" or profiles["VCM-Runtime"]["status"] != "GREEN":
        trigger_state = "YELLOW"
    else:
        trigger_state = "GREEN"
    not_claimed = [
        "native KV-aware runtime scheduling beyond hardware-aware route descriptors",
        "post-repair public LongMemEval confirmation beyond the private residual curriculum",
        "full public memory benchmark coverage for harder blocked families beyond the admitted five-family exact-run slice",
        "LongMemEval-V2 text/evaluator coverage until official non-image payload rows are staged locally",
    ]
    notes = [
        "The packet is a conceptual architecture packet; Theseus implements a local semantic/context compiler subset.",
        "Public benchmark content remains calibration-only and is not admitted to training rows.",
    ]
    if native_runtime_claimable:
        not_claimed.append("MLX-LM/CUDA/Metal-specific native KV parity beyond the exact backend reported by native_runtime_claim_scope")
        if not summary.get("scheduler_native_kv_route_allowed_for_recommended_backend"):
            not_claimed.append("scheduler-native KV routing for the recommended execution backend until that exact backend has a lifecycle proof")
        if summary.get("recommended_backend_runtime_descriptor_lifecycle_claimable"):
            notes.insert(
                2,
                "The recommended MLX backend now has a scoped mlx.core resident tensor descriptor lifecycle proof under complete VCM keys. This allows descriptor-level scheduling evidence for MLX while native model KV routing remains fail-closed until an MLX model runtime cache adapter exists.",
            )
        notes.insert(
            1,
            "VCM-Runtime has semantic runtime-key lifecycle evidence, hardware route metadata, and a local no-download Transformers/Torch tiny-model DynamicCache forward-pass KV reuse/invalidation proof under complete ordered VCM keys. The native KV claim is scoped to the exact reported backend and does not imply MLX/CUDA/Metal KV parity or scheduler-native routing for a different recommended backend.",
        )
    else:
        not_claimed.insert(0, "native KV/prefix cache lifecycle integration")
        not_claimed.insert(1, "native runtime/KV resident-materialization promises beyond the semantic lifecycle proof")
        notes.insert(
            1,
            "VCM-Runtime has semantic runtime-key lifecycle evidence and hardware route metadata; native KV/prefix reuse remains unclaimed until a local model-runtime lifecycle test passes.",
        )
    return {
        "policy": "project_theseus_vcm_release_conformance_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "packet": {
            "source": rel(packet_source),
            "release_audit": rel(packet_audit),
            "detected_terms": {
                "vcm_core": "VCM-Core" in packet_text,
                "vcm_governed": "VCM-Governed" in packet_text,
                "vcm_transactional": "VCM-Transactional" in packet_text,
                "vcm_predictive": "VCM-Predictive" in packet_text,
                "vcm_runtime": "VCM-Runtime" in packet_text,
                "unsafe_fit": "UNSAFE-FIT" in packet_text,
                "conformance_invariants": "VCM Invariants Checklist" in packet_text,
            },
        },
        "profiles": profiles,
        "invariants": invariants,
        "hard_failures": hard_failures,
        "not_claimed": not_claimed,
        "notes": notes,
    }


def derive_checks(
    *,
    packet_text: str,
    audit_text: str,
    compiled: dict[str, Any],
    probe: dict[str, Any],
    bench: dict[str, Any],
    training: dict[str, Any],
    consumers: dict[str, Any],
    recovery: dict[str, Any],
    graph: dict[str, Any],
    snapshots: dict[str, Any],
    pages: list[dict[str, Any]],
    prefetch_regret: dict[str, Any],
    runtime_readiness: dict[str, Any],
    runtime_lifecycle: dict[str, Any],
    native_runtime_probe: dict[str, Any],
    task_context_bridge: dict[str, Any],
    public_memory_calibration: dict[str, Any],
    public_memory_readiness: dict[str, Any],
    public_memory_prompt_calibration: dict[str, Any],
    public_memory_private_repair: dict[str, Any],
    private_lme_residual: dict[str, Any],
    vcm_evidence_gauntlet: dict[str, Any],
) -> dict[str, Any]:
    visible = list_value(compiled.get("model_visible_pages"))
    staged = list_value(compiled.get("staging_cache"))
    faults = list_value(compiled.get("semantic_page_faults"))
    protected_minimum = list_value(compiled.get("protected_minimum_set"))
    invariants = dict_value(compiled.get("invariants"))
    invalidation = dict_value(graph.get("invalidation"))
    relation_types = {str(row.get("type") or "") for row in list_value(graph.get("edges")) if isinstance(row, dict)}
    snapshot_rows = list_value(snapshots.get("snapshots"))
    page_certificate_failure_count = certificate_failure_count(pages)
    scoped_preference_pages = [
        page
        for page in pages
        if str(page.get("type") or "") == "scoped_preference"
        or str(page.get("execution_class") or "") == "scoped_user_preference"
    ]
    compile_has_observable_forecast = all(
        row.get("address")
        and "expected_value" in row
        and "deadline_step" in row
        and "probability" in row
        for row in list_value(compiled.get("context_demand_forecast"))[:20]
    )
    prefetch_summary = probe_summary(prefetch_regret)
    runtime_summary = probe_summary(runtime_readiness)
    runtime_lifecycle_summary = probe_summary(runtime_lifecycle)
    native_runtime_summary = probe_summary(native_runtime_probe)
    task_context_summary = probe_summary(task_context_bridge)
    public_memory_summary = probe_summary(public_memory_calibration)
    public_counters = dict_value(public_memory_summary.get("public_payload_counters"))
    public_readiness_gates = [
        row for row in list_value(public_memory_readiness.get("gates")) if isinstance(row, dict)
    ]
    public_readiness_contamination = dict_value(public_memory_readiness.get("contamination_counters"))
    public_readiness_private_summary = dict_value(public_memory_readiness.get("private_analogue_summary"))
    public_readiness_coverage = dict_value(public_memory_readiness.get("private_vcm_coverage"))
    public_prompt_summary = probe_summary(public_memory_prompt_calibration)
    public_prompt_per_benchmark = dict_value(public_prompt_summary.get("per_benchmark"))
    public_prompt_boundary = dict_value(public_memory_prompt_calibration.get("public_boundary"))
    public_prompt_lme = dict_value(public_prompt_per_benchmark.get("longmemeval"))
    public_prompt_lme_systems = dict_value(public_prompt_lme.get("memory_systems"))
    public_prompt_lme_best_single_non_vcm = max(
        [
            float(row.get("pass_rate") or 0.0)
            for name, row in public_prompt_lme_systems.items()
            if name != "vcm_graph_evidence_selector" and isinstance(row, dict)
        ],
        default=0.0,
    )
    public_prompt_ladder_confirmation = (
        int(public_prompt_summary.get("scored_item_count") or 0) >= 1000
        and {"8k_to_32k", "32k_to_128k", "128k_plus"}.issubset(set(dict_value(public_prompt_summary.get("per_length_bucket"))))
    )
    public_prompt_lme_repair_confirmation = (
        int(public_prompt_summary.get("scored_item_count") or 0) >= 600
        and int(public_prompt_lme.get("items") or 0) >= 200
        and float(public_prompt_lme.get("vcm_on_pass_rate") or 0.0) > 0.055
        and float(public_prompt_lme.get("vcm_on_pass_rate") or 0.0) > float(public_prompt_lme.get("vcm_off_pass_rate") or 0.0)
        and float(public_prompt_lme.get("vcm_on_pass_rate") or 0.0) >= public_prompt_lme_best_single_non_vcm
    )
    public_prompt_three_family_confirmation = (
        {"ruler", "babilong", "longmemeval"}.issubset(set(public_prompt_per_benchmark))
        and int(dict_value(public_prompt_per_benchmark.get("ruler")).get("items") or 0) > 0
        and int(dict_value(public_prompt_per_benchmark.get("babilong")).get("items") or 0) > 0
        and int(public_prompt_lme.get("items") or 0) > 0
        and (public_prompt_ladder_confirmation or public_prompt_lme_repair_confirmation)
    )
    public_prompt_largest_admitted_confirmation = (
        int(public_prompt_summary.get("scored_item_count") or 0) >= 2000
        and {"ruler", "babilong"}.issubset(set(public_prompt_per_benchmark))
        and int(dict_value(public_prompt_per_benchmark.get("ruler")).get("items") or 0) >= 1000
        and int(dict_value(public_prompt_per_benchmark.get("babilong")).get("items") or 0) >= 500
        and public_prompt_ladder_confirmation
        and len(dict_value(public_prompt_summary.get("per_length_bucket"))) >= 4
    )
    public_prompt_five_family_confirmation = (
        int(public_prompt_summary.get("scored_item_count") or 0) >= 2000
        and {
            "ruler",
            "babilong",
            "infinitebench",
            "needlebench_opencompass",
            "longbench_v2",
        }.issubset(set(public_prompt_per_benchmark))
        and all(
            int(dict_value(public_prompt_per_benchmark.get(benchmark)).get("items") or 0) > 0
            for benchmark in [
                "ruler",
                "babilong",
                "infinitebench",
                "needlebench_opencompass",
                "longbench_v2",
            ]
        )
        and len(dict_value(public_prompt_summary.get("per_length_bucket"))) >= 4
    )
    prefetch_regret_clean = (
        prefetch_regret.get("policy") == "project_theseus_vcm_prefetch_regret_audit_v1"
        and prefetch_regret.get("trigger_state") == "GREEN"
        and int(prefetch_summary.get("external_inference_calls") or 0) == 0
        and int(prefetch_summary.get("public_training_rows_written") or 0) == 0
        and int(prefetch_summary.get("fallback_return_count") or 0) == 0
        and bool(prefetch_summary.get("decision_time_features_complete"))
    )
    runtime_keys_ready = (
        runtime_readiness.get("policy") == "project_theseus_vcm_runtime_claim_readiness_v1"
        and runtime_readiness.get("trigger_state") == "GREEN"
        and runtime_summary.get("runtime_profile_claimed") is False
        and runtime_summary.get("native_kv_cache_claimed") is False
        and float(runtime_summary.get("cache_key_complete_rate") or 0.0) >= 1.0
        and int(runtime_summary.get("accepted_semantic_claims") or 0) > 0
        and int(runtime_summary.get("rejected_semantic_claims") or 0) == 0
    )
    runtime_lifecycle_clean = (
        runtime_lifecycle.get("policy") == "project_theseus_vcm_runtime_cache_lifecycle_v1"
        and runtime_lifecycle.get("trigger_state") == "GREEN"
        and int(runtime_lifecycle_summary.get("accepted_claims") or 0) > 0
        and float(runtime_lifecycle_summary.get("cache_key_complete_rate") or 0.0) >= 1.0
        and float(runtime_lifecycle_summary.get("reuse_hit_rate") or 0.0) >= 1.0
        and float(runtime_lifecycle_summary.get("snapshot_invalidation_miss_rate") or 0.0) >= 1.0
        and float(runtime_lifecycle_summary.get("policy_invalidation_miss_rate") or 0.0) >= 1.0
        and int(runtime_lifecycle_summary.get("cache_key_collision_count") or 0) == 0
        and runtime_lifecycle_summary.get("runtime_profile_claimed") is False
        and runtime_lifecycle_summary.get("native_kv_cache_claimed") is False
        and int(runtime_lifecycle_summary.get("external_inference_calls") or 0) == 0
        and int(runtime_lifecycle_summary.get("public_training_rows_written") or 0) == 0
        and int(runtime_lifecycle_summary.get("fallback_return_count") or 0) == 0
    )
    native_runtime_probe_clean = (
        native_runtime_probe.get("policy") == "project_theseus_vcm_native_runtime_probe_v1"
        and native_runtime_probe.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(native_runtime_summary.get("external_inference_calls") or 0) == 0
        and int(native_runtime_summary.get("public_training_rows_written") or 0) == 0
        and int(native_runtime_summary.get("fallback_return_count") or 0) == 0
        and int(native_runtime_summary.get("teacher_calls") or 0) == 0
    )
    native_runtime_claimable = (
        native_runtime_probe_clean
        and native_runtime_probe.get("trigger_state") == "GREEN"
        and native_runtime_summary.get("native_runtime_claimable") is True
        and native_runtime_summary.get("runtime_profile_claimed") is True
        and native_runtime_summary.get("native_kv_cache_claimed") is True
        and native_runtime_summary.get("native_prefix_cache_claimed") is True
        and native_runtime_summary.get("native_prefix_kv_lifecycle_test_passed") is True
    )
    hardware_route_metadata_ready = (
        native_runtime_probe_clean
        and native_runtime_summary.get("hardware_aware_runtime_route_metadata_ready") is True
        and int(native_runtime_summary.get("runtime_route_descriptor_count") or 0) > 0
    )
    task_context_bridge_clean = (
        task_context_bridge.get("policy") == "project_theseus_vcm_task_context_bridge_v1"
        and task_context_bridge.get("trigger_state") == "GREEN"
        and int(task_context_summary.get("task_family_count") or 0) >= 8
        and int(task_context_summary.get("high_priority_ready_count") or 0)
        == int(task_context_summary.get("high_priority_task_family_count") or 0)
        and int(task_context_summary.get("public_training_rows_written") or 0) == 0
        and int(task_context_summary.get("external_inference_calls") or 0) == 0
        and int(task_context_summary.get("fallback_return_count") or 0) == 0
        and task_context_summary.get("runtime_profile_claimed") is False
    )
    public_memory_clean = (
        public_memory_calibration.get("policy") == "project_theseus_vcm_public_memory_calibration_v1"
        and public_memory_calibration.get("calibration_mode") == "metadata_clean_public_benchmark_card_slice"
        and int(public_memory_summary.get("external_inference_calls") or 0) == 0
        and int(public_memory_summary.get("public_training_rows_written") or 0) == 0
        and int(public_memory_summary.get("fallback_return_count") or 0) == 0
        and all(int(value or 0) == 0 for value in public_counters.values())
    )
    public_memory_readiness_clean = (
        public_memory_readiness.get("policy") == "project_theseus_vcm_public_memory_readiness_audit_v1"
        and public_memory_readiness.get("trigger_state") == "GREEN"
        and public_memory_readiness.get("public_calibration_allowed") is True
        and bool(public_readiness_gates)
        and all(row.get("passed") is True for row in public_readiness_gates if row.get("severity") == "blocker")
        and all(int(value or 0) == 0 for value in public_readiness_contamination.values())
        and int(public_readiness_private_summary.get("off_only_wins") or 0) == 0
        and float(public_readiness_private_summary.get("vcm_on_pass_rate") or 0.0) >= 1.0
        and public_readiness_coverage.get("required_public_memory_residual_categories_present") is True
        and float(public_readiness_coverage.get("ablation_answer_lift") or 0.0) > 0.0
    )
    public_memory_prompt_clean = (
        public_memory_prompt_calibration.get("policy") == "project_theseus_vcm_public_memory_prompt_calibration_v1"
        and public_memory_prompt_calibration.get("trigger_state") in {"GREEN", "YELLOW"}
        and public_memory_prompt_calibration.get("calibration_mode") == "prompt_level_public_memory_quarantined_slice"
        and (
            public_prompt_three_family_confirmation
            or public_prompt_largest_admitted_confirmation
            or public_prompt_five_family_confirmation
        )
        and all(int(value or 0) == 0 for value in dict_value(public_prompt_summary.get("forbidden_overlap_counts")).values())
        and public_prompt_boundary.get("public_payloads_quarantined") is True
        and bool(get_path(public_memory_prompt_calibration, ["quarantine", "item_manifest"]))
        and bool(get_path(public_memory_prompt_calibration, ["quarantine", "item_manifest_hash"]))
        and public_prompt_boundary.get("public_training_use_allowed") is False
        and int(public_prompt_boundary.get("public_rows_admitted_to_training") or 0) == 0
        and int(public_prompt_summary.get("external_inference_calls") or 0) == 0
        and int(public_prompt_summary.get("public_training_rows_written") or 0) == 0
        and int(public_prompt_summary.get("fallback_return_count") or 0) == 0
        and int(public_prompt_summary.get("teacher_solving_calls") or 0) == 0
    )
    public_memory_private_repair_clean = (
        public_memory_private_repair.get("policy") == "project_theseus_vcm_public_memory_private_residual_repair_v1"
        and public_memory_private_repair.get("private_only") is True
        and int(public_memory_private_repair.get("public_prompt_chars") or 0) == 0
        and int(public_memory_private_repair.get("public_answer_chars") or 0) == 0
        and int(public_memory_private_repair.get("public_training_rows_written") or 0) == 0
        and int(public_memory_private_repair.get("external_inference_calls") or 0) == 0
        and int(public_memory_private_repair.get("fallback_return_count") or 0) == 0
        and (
            int(public_memory_private_repair.get("fixture_count") or 0) > 0
            or (
                public_memory_private_repair.get("repair_needed") is False
                and int(public_memory_private_repair.get("fixture_count") or 0) == 0
            )
        )
    )
    private_lme_summary = probe_summary(private_lme_residual)
    private_lme_proposal = dict_value(private_lme_residual.get("future_public_calibration_proposal"))
    vcm_evidence_gauntlet_summary = probe_summary(vcm_evidence_gauntlet)
    vcm_evidence_gauntlet_abstention = dict_value(vcm_evidence_gauntlet_summary.get("abstention"))
    vcm_evidence_gauntlet_proposal = dict_value(vcm_evidence_gauntlet.get("public_confirmation_manifest_proposal"))
    private_lme_residual_clean = (
        private_lme_residual.get("policy") == "project_theseus_vcm_longmemeval_private_residual_curriculum_v1"
        and private_lme_residual.get("trigger_state") == "GREEN"
        and private_lme_residual.get("private_only") is True
        and int(private_lme_summary.get("case_count") or 0) >= 150
        and float(private_lme_summary.get("vcm_on_pass_rate") or 0.0) >= 0.85
        and float(private_lme_summary.get("minimum_major_question_type_pass_rate") or 0.0) >= 0.75
        and float(private_lme_summary.get("vcm_over_best_single_non_vcm_delta") or 0.0) >= 0.05
        and float(private_lme_summary.get("vcm_on_evidence_recall") or 0.0) >= 0.85
        and private_lme_proposal.get("proposal_state") == "READY_TO_PROPOSE_EXACT_ONCE_PUBLIC_CONFIRMATION"
        and private_lme_proposal.get("run_public_automatically") is False
        and len(list_value(private_lme_residual.get("hard_failures"))) == 0
        and all(
            int(private_lme_residual.get(key) or 0) == 0
            for key in [
                "external_inference_calls",
                "teacher_solving_calls",
                "fallback_return_count",
                "public_training_rows_written",
                "public_prompt_chars_loaded",
                "public_context_chars_loaded",
                "public_answer_chars_loaded",
            ]
        )
    )
    vcm_evidence_gauntlet_clean = (
        vcm_evidence_gauntlet.get("policy") == "project_theseus_vcm_evidence_gauntlet_v1"
        and vcm_evidence_gauntlet.get("trigger_state") == "GREEN"
        and vcm_evidence_gauntlet.get("private_only") is True
        and int(vcm_evidence_gauntlet_summary.get("case_count") or 0) >= 1000
        and float(vcm_evidence_gauntlet_summary.get("vcm_on_pass_rate") or 0.0) >= 0.90
        and float(vcm_evidence_gauntlet_summary.get("minimum_major_family_pass_rate") or 0.0) >= 0.80
        and float(vcm_evidence_gauntlet_summary.get("vcm_over_best_single_non_vcm_delta") or 0.0) >= 0.05
        and float(vcm_evidence_gauntlet_summary.get("vcm_on_evidence_recall") or 0.0) >= 0.90
        and float(vcm_evidence_gauntlet_abstention.get("precision") or 0.0) >= 0.95
        and float(vcm_evidence_gauntlet_abstention.get("recall") or 0.0) >= 0.95
        and len(list_value(vcm_evidence_gauntlet.get("hard_failures"))) == 0
        and vcm_evidence_gauntlet_proposal.get("run_public_automatically") is False
        and all(
            int(vcm_evidence_gauntlet.get(key) or vcm_evidence_gauntlet_summary.get(key) or 0) == 0
            for key in [
                "external_inference_calls",
                "teacher_solving_calls",
                "fallback_return_count",
                "public_training_rows_written",
                "public_prompt_chars_loaded",
                "public_context_chars_loaded",
                "public_answer_chars_loaded",
                "public_trace_chars_loaded",
                "public_test_chars_loaded",
                "public_solution_chars_loaded",
                "public_template_chars_loaded",
            ]
        )
    )
    public_memory_prompt_no_regression = (
        public_memory_prompt_clean
        and public_memory_prompt_positive_transfer(public_prompt_summary)
        and (
            int(get_path(public_prompt_summary, ["win_counts", "vcm_off"]) or 0) == 0
            or public_memory_private_repair_clean
        )
    )
    return {
        "packet_present": bool(packet_text) and bool(audit_text),
        "packet_conceptual": "conceptual architecture" in packet_text.lower()
        and "no implementation results" in audit_text.lower(),
        "probe_green": probe.get("trigger_state") == "GREEN",
        "bench_green": bench.get("trigger_state") == "GREEN",
        "training_green": training.get("trigger_state") == "GREEN",
        "consumer_green": consumers.get("trigger_state") == "GREEN",
        "recovery_green": recovery.get("trigger_state") == "GREEN",
        "stable_visible_versions": all(str(row.get("address") or "").startswith("vcm://") and "@v" in str(row.get("address") or "") for row in visible),
        "visible_have_certificates": all(row.get("certificate_id") for row in visible),
        "page_certificates_complete": page_certificate_failure_count == 0,
        "named_snapshot": bool(compiled.get("snapshot")) and bool(snapshot_rows),
        "read_your_writes": any(bool(dict_value(row).get("read_your_writes")) for row in snapshot_rows),
        "conflicts_represented": {"contradicts", "supersedes", "rejected_because"}.issubset(relation_types),
        "staging_non_influential": bool(staged) and all(row.get("non_influential") is True for row in staged),
        "promotion_gates_proxy": invariants.get("all_model_visible_pages_have_certificates") is True
        and invariants.get("faults_are_explicit") is True
        and gate_passed(probe, "tainted_instruction_promotion_blocked")
        and gate_passed(probe, "stale_current_claim_faults")
        and gate_passed(probe, "exactness_gap_faults"),
        "tool_auth_independent_partial": True,
        "runtime_cache_not_claimed": True,
        "deletion_closure": bool(invalidation.get("deletion_closure_complete")) and bool(invalidation.get("cache_invalidation_records")),
        "scoped_preference_controls": bool(scoped_preference_pages) or bench_case_passed(bench, "over_personalization_restraint"),
        "context_switch_snapshots": len(snapshot_rows) >= 2 and all(dict_value(row).get("mount_table") for row in snapshot_rows),
        "faults_safe": all(row.get("safe_behavior") == "do_not_reconstruct_missing_context" and row.get("fallback_path") for row in faults),
        "authority_non_escalation": page_certificate_failure_count == 0
        and all(
            dict_value(dict_value(rep).get("certificate")).get("authority_ceiling")
            for page in pages
            for rep in dict_value(page.get("representations")).values()
            if isinstance(rep, dict)
        ),
        "unsafe_fit_explicit": invariants.get("protected_minimum_fit_or_explicit_unsafe_fit") is True
        and dict_value(compiled.get("unsafe_fit_result")).get("explicit") is True
        and isinstance(compiled.get("unsafe_fit"), bool)
        and bool(protected_minimum),
        "lossy_not_exact": all(
            not (
                dict_value(rep).get("lossy") is True
                and "exact_wording" in set(dict_value(rep).get("intended_uses") or [])
            )
            for page in pages
            for rep in dict_value(page.get("representations")).values()
            if isinstance(rep, dict)
        ),
        "deterministic_constraints": invariants.get("no_external_inference") is True
        and invariants.get("no_public_calibration") is True
        and int(probe_summary(probe).get("fallback_return_count") or 0) == 0,
        "deletion_not_frequency_only": bool(invalidation.get("tombstone_event")),
        "protected_admission_external": bool(protected_minimum)
        and all(row.get("authorized") is True for row in protected_minimum),
        "runtime_prefix_not_claimed": True,
        "decision_time_features": compile_has_observable_forecast,
        "resident_materialization_not_claimed": True,
        "predictive_forecast_present": compile_has_observable_forecast,
        "prefetch_regret_accounting": prefetch_regret_clean,
        "runtime_cache_key_complete": runtime_keys_ready,
        "runtime_cache_lifecycle_clean": runtime_lifecycle_clean,
        "native_runtime_probe_clean": native_runtime_probe_clean,
        "native_runtime_claimable": native_runtime_claimable,
        "hardware_route_metadata_ready": hardware_route_metadata_ready,
        "resident_materialization_claims_ready": runtime_keys_ready,
        "task_context_bridge_clean": task_context_bridge_clean,
        "public_memory_calibration_clean": public_memory_clean,
        "public_memory_readiness_clean": public_memory_readiness_clean,
        "public_memory_prompt_calibration_clean": public_memory_prompt_clean,
        "public_memory_prompt_no_regression": public_memory_prompt_no_regression,
        "public_memory_private_repair_clean": public_memory_private_repair_clean,
        "private_longmemeval_residual_clean": private_lme_residual_clean,
        "vcm_evidence_gauntlet_clean": vcm_evidence_gauntlet_clean,
        "external_inference_zero": total_external_inference(compiled, probe, bench, training, recovery) == 0,
        "public_training_zero": int(probe_summary(probe).get("public_training_rows_written") or 0) == 0,
        "public_calibration_zero": int(probe_summary(probe).get("public_calibration_runs") or 0) == 0,
        "fallback_zero": int(probe_summary(probe).get("fallback_return_count") or 0) == 0,
    }


def build_profiles(checks: dict[str, Any]) -> dict[str, dict[str, Any]]:
    core_checks = [
        "packet_present",
        "stable_visible_versions",
        "visible_have_certificates",
        "page_certificates_complete",
        "promotion_gates_proxy",
        "faults_safe",
        "authority_non_escalation",
        "unsafe_fit_explicit",
        "probe_green",
        "bench_green",
        "recovery_green",
    ]
    governed_checks = [
        *core_checks,
        "consumer_green",
        "task_context_bridge_clean",
        "training_green",
        "deletion_closure",
        "scoped_preference_controls",
        "external_inference_zero",
        "public_training_zero",
        "fallback_zero",
        "public_memory_readiness_clean",
        "public_memory_prompt_calibration_clean",
        "public_memory_private_repair_clean",
        "private_longmemeval_residual_clean",
        "vcm_evidence_gauntlet_clean",
    ]
    transactional_checks = [
        *core_checks,
        "named_snapshot",
        "read_your_writes",
        "context_switch_snapshots",
        "conflicts_represented",
        "deletion_closure",
    ]
    predictive_checks = [
        *core_checks,
        "predictive_forecast_present",
        "staging_non_influential",
        "decision_time_features",
        "prefetch_regret_accounting",
    ]
    return {
        "VCM-Core": profile("VCM-Core", core_checks, checks),
        "VCM-Governed": profile("VCM-Governed", governed_checks, checks),
        "VCM-Transactional": profile("VCM-Transactional", transactional_checks, checks),
        "VCM-Predictive": profile(
            "VCM-Predictive",
            predictive_checks,
            checks,
            yellow_when_missing={"prefetch_regret_accounting"},
            note="Forecasting, non-model-visible staging, and private prefetch-regret accounting are required before Predictive VCM is green.",
        ),
        "VCM-Runtime": {
            "status": (
                "GREEN"
                if checks.get("native_runtime_claimable")
                else "YELLOW"
                if checks.get("runtime_cache_key_complete") and checks.get("runtime_cache_lifecycle_clean") and checks.get("native_runtime_probe_clean")
                else "NOT_CLAIMED"
            ),
            "evidence": (
                "Native prefix/KV cache lifecycle is proven with a local tiny-model forward-pass DynamicCache under complete ordered VCM runtime keys."
                if checks.get("native_runtime_claimable")
                else "Semantic runtime-key lifecycle and hardware route metadata are proven; native prefix/KV cache reuse is not claimed."
                if checks.get("runtime_cache_key_complete") and checks.get("runtime_cache_lifecycle_clean") and checks.get("native_runtime_probe_clean")
                else "Native prefix/KV cache lifecycle is not implemented; semantic runtime-key readiness is tracked separately."
            ),
            "checks": {
                "semantic_runtime_cache_key_complete": bool(checks.get("runtime_cache_key_complete")),
                "semantic_runtime_cache_lifecycle_clean": bool(checks.get("runtime_cache_lifecycle_clean")),
                "semantic_resident_materialization_claims_ready": bool(checks.get("resident_materialization_claims_ready")),
                "native_runtime_probe_clean": bool(checks.get("native_runtime_probe_clean")),
                "native_kv_prefix_cache_claimed": bool(checks.get("native_runtime_claimable")),
                "hardware_aware_runtime_route_metadata": bool(checks.get("hardware_route_metadata_ready")),
            },
        },
    }


def profile(
    name: str,
    required: list[str],
    checks: dict[str, Any],
    *,
    yellow_when_missing: set[str] | None = None,
    note: str = "",
) -> dict[str, Any]:
    yellow_when_missing = yellow_when_missing or set()
    failed = [key for key in required if not checks.get(key)]
    hard_failed = [key for key in failed if key not in yellow_when_missing]
    if hard_failed:
        status = "RED"
    elif failed:
        status = "YELLOW"
    else:
        status = "GREEN"
    return {
        "status": status,
        "evidence": f"{name}: failed={failed}" if failed else f"{name}: all required checks passed",
        "checks": {key: bool(checks.get(key)) for key in required},
        "note": note,
    }


def build_invariants(checks: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = {
        1: ("GREEN" if checks["stable_visible_versions"] else "RED", "model-visible addresses are immutable vcm://...@v versions", True),
        2: ("GREEN" if checks["page_certificates_complete"] else "RED", "representations expose certificates, authority ceilings, declared loss, and fallback paths", True),
        3: ("GREEN" if checks["named_snapshot"] and checks["read_your_writes"] else "RED", "snapshot and read-your-writes records are present", True),
        4: ("GREEN" if checks["conflicts_represented"] else "YELLOW", "semantic graph records contradiction/supersession/rejection fixtures", False),
        5: ("GREEN" if checks["staging_non_influential"] else "RED", "staged pages are explicitly non-model-visible", True),
        6: ("GREEN" if checks["promotion_gates_proxy"] else "RED", "promotion gate probes cover capability, taint, freshness, exactness, and proof checks", True),
        7: ("YELLOW", "Project-level tool policy exists, but VCM is not yet the direct tool-action authorization boundary", False),
        8: (
            "GREEN" if checks["native_runtime_claimable"] else ("YELLOW" if checks["runtime_cache_key_complete"] else "NOT_CLAIMED"),
            "native runtime cache keys passed local tiny-model forward-pass lifecycle tests"
            if checks["native_runtime_claimable"]
            else "semantic materialization keys are complete; native runtime cache-key parity is not claimed"
            if checks["runtime_cache_key_complete"]
            else "VCM-Runtime cache-key completeness is not claimed",
            False,
        ),
        9: ("GREEN" if checks["deletion_closure"] else "RED", "graph invalidation includes descendant/cache records", True),
        10: ("GREEN" if checks["scoped_preference_controls"] else "YELLOW", "scoped preference probe blocks preference-to-policy escalation", False),
        11: ("GREEN" if checks["context_switch_snapshots"] else "RED", "multi-snapshot mount tables are present", True),
        12: ("GREEN" if checks["faults_safe"] else "RED", "faults include safe behavior and source fallback", True),
        13: ("GREEN" if checks["authority_non_escalation"] else "RED", "certificates carry authority ceilings and memory text cannot authorize tools", True),
        14: ("GREEN" if checks["unsafe_fit_explicit"] else "RED", "compiled context exposes protected minimum set and explicit unsafe-fit result", True),
        15: ("GREEN" if checks["lossy_not_exact"] else "RED", "lossy representations are not admitted as exact wording", True),
        16: ("GREEN" if checks["deterministic_constraints"] else "RED", "learned/forecast decisions are constrained by deterministic no-cheat gates", True),
        17: ("GREEN" if checks["deletion_not_frequency_only"] else "YELLOW", "deletion is tombstone/closure-driven, not frequency-only", False),
        18: ("GREEN" if checks["protected_admission_external"] else "RED", "protected minimum rows are compiler-authorized, not page self-pinned", True),
        19: (
            "GREEN" if checks["native_runtime_claimable"] else "NOT_CLAIMED",
            "native reusable prefix/KV objects passed complete-key local forward-pass reuse/invalidation tests"
            if checks["native_runtime_claimable"]
            else "native reusable prefix/KV objects are not claimed",
            False,
        ),
        20: ("GREEN" if checks["decision_time_features"] else "YELLOW", "forecast rows expose decision-time probability/deadline/value features", False),
        21: (
            "GREEN" if checks["native_runtime_claimable"] else ("YELLOW" if checks["resident_materialization_claims_ready"] else "NOT_CLAIMED"),
            "native future-reuse promise passed complete-key tiny-model runtime lifecycle tests"
            if checks["native_runtime_claimable"]
            else "semantic resident-materialization descriptors are ready; runtime future-reuse promises are not claimed"
            if checks["resident_materialization_claims_ready"]
            else "resident-materialization promises are not claimed",
            False,
        ),
    }
    rows = []
    for idx in range(1, 22):
        status, evidence, required = mapping[idx]
        rows.append(
            {
                "id": idx,
                "status": status,
                "required_for_core": required,
                "text": INVARIANT_TEXT[idx],
                "evidence": evidence,
            }
        )
    return rows


def gate_passed(report: dict[str, Any], name: str) -> bool:
    for row in list_value(report.get("gates")):
        if isinstance(row, dict) and row.get("gate") == name:
            return row.get("passed") is True
    return False


def bench_case_passed(report: dict[str, Any], name: str) -> bool:
    for row in list_value(report.get("cases")):
        if isinstance(row, dict) and row.get("id") == name:
            return row.get("vcm_passed") is True
    return False


def certificate_failure_count(pages: list[dict[str, Any]]) -> int:
    failures = 0
    for page in pages:
        for level, rep in dict_value(page.get("representations")).items():
            if not isinstance(rep, dict):
                failures += 1
                continue
            cert = dict_value(rep.get("certificate"))
            if not cert:
                failures += 1
                continue
            if level in {"L1", "L2", "L3", "L4"} and not cert.get("certificate_id"):
                failures += 1
            if not cert.get("fallback_path"):
                failures += 1
            if not dict_value(cert.get("authority_ceiling")):
                failures += 1
            if not get_path(cert, ["declared_loss", "fallback_path"]):
                failures += 1
    return failures


def total_external_inference(*reports: dict[str, Any]) -> int:
    total = 0
    for report in reports:
        total += int(report.get("external_inference_calls") or 0)
        total += int(probe_summary(report).get("external_inference_calls") or 0)
    return total


def probe_summary(report: dict[str, Any]) -> dict[str, Any]:
    return dict_value(report.get("summary"))


def public_memory_prompt_positive_transfer(summary: dict[str, Any]) -> bool:
    scored = int(summary.get("scored_item_count") or 0)
    vcm_on = float(summary.get("vcm_on_pass_rate") or 0.0)
    vcm_off = float(summary.get("vcm_off_pass_rate") or 0.0)
    off_only = int(get_path(summary, ["win_counts", "vcm_off"]) or 0)
    delta = float(summary.get("vcm_over_flat_tail_delta") or (vcm_on - vcm_off))
    best_delta = float(summary.get("vcm_over_best_non_vcm_delta") or 0.0)
    if scored >= 100:
        return vcm_on > vcm_off and delta > 0.0 and best_delta > 0.0
    return vcm_on >= vcm_off and off_only == 0


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VCM Release Conformance Audit",
        "",
        f"State: `{report['trigger_state']}`",
        "",
        "## Summary",
        "",
        f"- Core profiles ready: `{summary['core_profiles_ready']}`",
        f"- Hard failures: `{summary['hard_failure_count']}`",
        f"- Runtime profile claimed: `{summary['runtime_profile_claimed']}`",
        f"- Native runtime probe: state `{summary.get('native_runtime_probe_state')}`, claimable `{summary.get('native_runtime_claimable')}`, native lifecycle `{summary.get('native_prefix_kv_lifecycle_test_passed')}`",
        f"- Native runtime route: metadata-ready `{summary.get('native_runtime_route_metadata_ready')}`, backend `{summary.get('native_runtime_recommended_backend')}`, Python `{summary.get('native_runtime_recommended_python')}`, blockers `{summary.get('native_runtime_blocker_count')}`",
        f"- Native runtime claim backend/recommended backend: `{summary.get('native_runtime_claim_backend')}` / `{summary.get('native_runtime_recommended_execution_backend')}`",
        f"- Scheduler native KV route allowed for recommended backend: `{summary.get('scheduler_native_kv_route_allowed_for_recommended_backend')}`",
        f"- External inference calls: `{summary['external_inference_calls']}`",
        f"- Public training rows written: `{summary['public_training_rows_written']}`",
        f"- Fallback return count: `{summary['fallback_return_count']}`",
        f"- Prompt public memory VCM-on pass rate: `{summary.get('public_memory_prompt_vcm_on_pass_rate')}`",
        f"- Prompt public memory VCM-off pass rate: `{summary.get('public_memory_prompt_vcm_off_pass_rate')}`",
        f"- Prompt public memory VCM over flat-tail delta: `{summary.get('public_memory_prompt_vcm_over_flat_tail_delta')}`",
        f"- Prompt public memory VCM over best non-VCM delta: `{summary.get('public_memory_prompt_vcm_over_best_non_vcm_delta')}`",
        f"- Prompt public memory best non-VCM: `{summary.get('public_memory_prompt_best_non_vcm')}`",
        f"- Prompt public memory off-only wins: `{summary.get('public_memory_prompt_off_only_wins')}`",
        f"- Prompt public memory source token distribution: `{summary.get('public_memory_prompt_source_context_token_distribution')}`",
        f"- Prompt public memory item manifest hash: `{summary.get('public_memory_prompt_item_manifest_hash')}`",
        f"- Prompt public memory no regression: `{summary.get('public_memory_prompt_no_regression')}`",
        f"- Public memory readiness clean: `{summary.get('public_memory_readiness_clean')}`",
        f"- VCM evidence gauntlet: state `{summary.get('vcm_evidence_gauntlet_state')}`, cases `{summary.get('vcm_evidence_gauntlet_cases')}`, VCM `{summary.get('vcm_evidence_gauntlet_vcm_pass_rate')}`, best non-VCM `{summary.get('vcm_evidence_gauntlet_best_single_non_vcm')}`, delta `{summary.get('vcm_evidence_gauntlet_delta')}`",
        f"- VCM evidence gauntlet min family: `{summary.get('vcm_evidence_gauntlet_min_family')}`",
        f"- VCM evidence gauntlet abstention: `{summary.get('vcm_evidence_gauntlet_abstention')}`",
        f"- Hard-memory private gauntlet: state `{summary.get('hard_memory_private_state')}`, cases `{summary.get('hard_memory_private_cases')}`, families `{summary.get('hard_memory_private_family_count')}`, buckets `{summary.get('hard_memory_private_length_bucket_count')}`, VCM `{summary.get('hard_memory_private_vcm_pass_rate')}`, best non-VCM `{summary.get('hard_memory_private_best_single_non_vcm')}`, delta `{summary.get('hard_memory_private_delta')}`",
        f"- Hard-memory readiness: state `{summary.get('hard_memory_readiness_state')}`, public rows `{summary.get('hard_memory_readiness_public_rows')}` / `{summary.get('hard_memory_readiness_public_target')}`, metadata-ready `{summary.get('hard_memory_readiness_metadata_ready_count')}`, blocked/queued `{summary.get('hard_memory_readiness_blocked_or_queued_count')}`",
        "",
        "## Profiles",
        "",
    ]
    for name, row in report["profiles"].items():
        lines.append(f"- `{name}`: `{row['status']}` - {row['evidence']}")
        if row.get("note"):
            lines.append(f"  - {row['note']}")
    lines.extend(["", "## Invariants", ""])
    for row in report["invariants"]:
        lines.append(f"- `{row['id']}` `{row['status']}`: {row['text']} Evidence: {row['evidence']}")
    lines.extend(["", "## Not Claimed", ""])
    for item in report["not_claimed"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def get_path(value: Any, path: list[str]) -> Any:
    cursor = value
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
