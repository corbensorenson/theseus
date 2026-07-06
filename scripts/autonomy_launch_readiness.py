"""Launch readiness checker for long SparkStream autonomy runs.

This script answers a narrow question: is the local project ready to start an
autonomous training/ratcheting session with the teacher available as a sparse
advisor? It does not launch training. It emits a machine-readable report for
the dashboard and for checkpoint artifacts.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import license_manager  # noqa: E402
import teacher_oracle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="inner_loop")
    parser.add_argument("--require-teacher-cli", action="store_true")
    parser.add_argument("--out", default="reports/autonomy_launch_readiness.json")
    args = parser.parse_args()

    reports = ROOT / "reports"
    configs = ROOT / "configs"
    preflight = read_json(reports / "training_preflight_report.json")
    macos_preflight = read_json(reports / "macos_training_preflight.json")
    candidate = read_json(reports / "candidate_promotion_gate.json")
    resource = read_json(reports / "resource_governor.json")
    arm_governance = read_json(reports / "arm_lifecycle_governance.json")
    teacher_policy = read_json(configs / "teacher_policy.json")
    teacher_cli_path = teacher_oracle.resolve_codex_command(teacher_policy)
    teacher_cli_available = codex_cli_available(teacher_cli_path)
    teacher_smoke = read_json(reports / "teacher_oracle_last.json")
    teacher_calls = read_jsonl(ROOT / str(teacher_policy.get("log_path", "reports/teacher_calls.jsonl")))
    autonomy_policy = read_json(configs / "autonomy_policy.json")
    checkpoint_backup_policy = read_json(configs / "checkpoint_backup_policy.json")
    checkpoint_backup = read_json(reports / "checkpoint_backup_last.json")
    checkpoint_registry = read_json(reports / "checkpoint_registry.json")
    external_audit = read_json(reports / "external_inference_audit.json")
    attd = read_json(reports / "attd_report.json")
    attd_packets = read_json(reports / "attd_maintenance_packets.json")
    autoresearch_audit = read_json(reports / "autoresearch_gap_audit.json")
    personality_context = read_json(reports / "personality_context_last.json")
    vcm_probe = read_json(reports / "virtual_context_memory_probe.json")
    vcm_status = read_json(reports / "virtual_context_memory_status.json")
    vcm_training = read_json(reports / "virtual_context_memory_training_admission.json")
    vcm_consumer_audit = read_json(reports / "virtual_context_memory_consumer_audit.json")
    vcm_task_context_bridge = read_json(reports / "vcm_task_context_bridge.json")
    vcm_task_contexts = read_json(reports / "vcm_task_contexts.json")
    vcm_context_recovery = read_json(reports / "vcm_context_recovery_benchmark.json")
    vcm_on_off_ablation = read_json(reports / "vcm_on_off_ablation.json")
    vcm_public_memory_calibration = read_json(reports / "vcm_public_memory_calibration.json")
    vcm_public_memory_readiness = read_json(reports / "vcm_public_memory_readiness_audit.json")
    vcm_public_memory_prompt_calibration = read_json(reports / "vcm_public_memory_prompt_calibration.json")
    vcm_public_memory_private_repair = read_json(reports / "vcm_public_memory_private_residual_repair.json")
    vcm_longmemeval_private_residual = read_json(reports / "vcm_longmemeval_private_residual_curriculum.json")
    vcm_evidence_gauntlet = read_json(reports / "vcm_evidence_gauntlet.json")
    vcm_hard_memory_private = read_json(reports / "vcm_hard_memory_private_analogues.json")
    vcm_hard_memory_readiness = read_json(reports / "vcm_hard_memory_benchmark_readiness.json")
    vcm_prefetch_regret = read_json(reports / "vcm_prefetch_regret_audit.json")
    vcm_runtime_claim_readiness = read_json(reports / "vcm_runtime_claim_readiness.json")
    vcm_release_conformance = read_json(reports / "vcm_release_conformance_audit.json")
    personality_drift = read_json(reports / "personality_drift_eval.json")
    personality_runtime = read_json(reports / "personality_runtime_audit.json")
    belief_governance = read_json(reports / "belief_update_governance.json")
    legacy_runtime = read_json(reports / "legacy_runtime_governance_gate.json")
    legacy_runtime_enforcement = read_json(reports / "legacy_port_runtime_enforcement.json")
    coherence_gate = read_json(reports / "coherence_delirium_gate.json")
    legacy_training_sources = read_json(reports / "legacy_training_source_audit.json")
    legacy_training_sample = read_json(reports / "legacy_training_source_sample.json")
    legacy_rl_envs = read_json(reports / "legacy_rl_environment_admission.json")
    legacy_rl_smoke = read_json(reports / "legacy_rl_smoke_plan.json")
    trace_capsules = read_json(reports / "trace_fabric_capsule_admission.json")
    trace_materialization = read_json(reports / "trace_fabric_capsule_materialization.json")
    legacy_adapter_plan = read_json(reports / "legacy_adapter_bank_training_plan.json")
    active_inference_pilot = read_json(reports / "legacy_active_inference_pilot.json")
    whitecell = read_json(reports / "whitecell_threat_memory.json")
    whitecell_remediation = read_json(reports / "whitecell_remediation.json")
    hive_policy = read_json(configs / "hive_policy.json")
    hive_status = read_json(reports / "hive_status.json")
    hive_peers = read_json(reports / "hive_peers.json")
    hive_scheduler = read_json(reports / "hive_scheduler.json")
    license_status = license_manager.status_report(write_report=True)
    license_worker_chunks = license_manager.check_feature("distributed_worker_chunks", write_report=True)
    teacher_smoke_passed = teacher_smoke_ok(teacher_smoke, teacher_calls)
    whitecell_blockers = active_whitecell_blockers(whitecell, whitecell_remediation)
    is_macos = platform.system() == "Darwin"
    local_macos_smoke_ready = (
        is_macos
        and args.profile == "smoke"
        and macos_preflight.get("policy") == "project_theseus_macos_training_preflight_v0"
        and bool(macos_preflight.get("bounded_smoke_allowed"))
        and int(get_path(macos_preflight, ["summary", "hard_failures"], 0) or 0) == 0
    )
    local_macos_training_ready = (
        is_macos
        and args.profile in {"inner_loop", "candidate", "overnight", "seed_sweep"}
        and macos_preflight.get("policy") == "project_theseus_macos_training_preflight_v0"
        and macos_preflight.get("state") == "GREEN"
        and bool(macos_preflight.get("long_training_allowed"))
        and int(get_path(macos_preflight, ["summary", "hard_failures"], 0) or 0) == 0
    )
    heavy_training_preflight_ready = (
        bool(preflight.get("heavy_training_allowed"))
        and int(preflight.get("blocker_count") or 0) == 0
    )
    preflight_ready = heavy_training_preflight_ready or local_macos_smoke_ready or local_macos_training_ready
    seed55_path = reports / "babylm_mutated_holdout_seed55_stateful_grammar_state_frontier.json"
    seed55_ready = seed55_path.exists() or local_macos_smoke_ready

    vcm_runtime_profile_claimed = get_path(vcm_release_conformance, ["summary", "runtime_profile_claimed"], True)
    vcm_release_runtime_claim_ok = (
        vcm_runtime_profile_claimed is False
        or (
            vcm_runtime_profile_claimed is True
            and get_path(vcm_release_conformance, ["summary", "native_runtime_claimable"], False) is True
            and get_path(vcm_release_conformance, ["summary", "native_prefix_kv_lifecycle_test_passed"], False) is True
            and int(get_path(vcm_release_conformance, ["summary", "native_runtime_blocker_count"], 1) or 0) == 0
        )
    )

    checks = [
        check(
            "external_inference_teacher_only",
            bool(external_audit.get("ok")) and external_audit.get("teacher_only_invariant") is True,
            "blocker",
            f"ok={external_audit.get('ok')} summary={external_audit.get('summary')}",
        ),
        check(
            "preflight_allows_heavy_training",
            preflight_ready,
            "blocker",
            (
                f"heavy_training_allowed={preflight.get('heavy_training_allowed')} "
                f"blockers={preflight.get('blocker_count')} "
                f"macos_bounded_smoke={local_macos_smoke_ready} "
                f"macos_long_training={local_macos_training_ready} "
                f"macos_state={macos_preflight.get('state')}"
            ),
        ),
        check(
            "candidate_gate_available",
            bool(candidate) and isinstance(candidate.get("checks"), list),
            "blocker",
            f"promote={candidate.get('promote')} passed={candidate.get('passed')} total={candidate.get('total')}",
        ),
        check(
            "seed55_frontier_artifact_exists",
            seed55_ready,
            "blocker",
            "seed55 mutated holdout report exists; Mac-local smoke can proceed without it.",
        ),
        check(
            "resource_governor_green",
            bool((resource.get("decision") or {}).get("can_run_requested_profile", True)),
            "blocker",
            f"profile={resource.get('requested_profile')} recommended={(resource.get('decision') or {}).get('recommended_profile')}",
        ),
        check(
            "arm_lifecycle_governance_ready",
            bool(arm_governance.get("ready_for_long_autonomy")),
            "blocker",
            f"arms={(arm_governance.get('summary') or {}).get('arms')} proposals={(arm_governance.get('summary') or {}).get('proposal_count')}",
        ),
        check(
            "attd_report_available",
            bool(attd) and attd.get("policy") == "sparkstream_attd_report_v0",
            "blocker",
            f"trigger_state={attd.get('trigger_state')} score={attd.get('attd_score')}",
        ),
        check(
            "attd_allows_long_autonomy",
            bool((attd.get("governance") or {}).get("allows_long_autonomy", False)),
            "blocker",
            f"trigger_state={attd.get('trigger_state')} packets={attd_packets.get('packet_count')}",
        ),
        check(
            "autoresearch_gap_audit_available",
            autoresearch_audit.get("policy") == "sparkstream_autoresearch_gap_audit_v0",
            "warning",
            f"state={(autoresearch_audit.get('summary') or {}).get('trigger_state')} gaps={(autoresearch_audit.get('summary') or {}).get('gap_count')}",
        ),
        check(
            "autoresearch_experiment_ledger_baselined",
            not bool((autoresearch_audit.get("summary") or {}).get("needs_baseline", True)),
            "warning",
            f"entries={(autoresearch_audit.get('summary') or {}).get('ledger_entries')}",
        ),
        check(
            "personality_context_ready",
            personality_context.get("status") == "ready"
            and int(get_path(personality_context, ["summary", "selected_cards"], 0) or 0) > 0
            and int(get_path(personality_context, ["summary", "hard_safety_invariants"], 0) or 0) > 0,
            "blocker",
            f"status={personality_context.get('status')} selected_cards={get_path(personality_context, ['summary', 'selected_cards'], None)} hard_invariants={get_path(personality_context, ['summary', 'hard_safety_invariants'], None)}",
        ),
        check(
            "personality_drift_eval_passed",
            personality_drift.get("passed") is True and float(get_path(personality_drift, ["summary", "average_score"], 0.0) or 0.0) >= 0.75,
            "blocker",
            f"passed={personality_drift.get('passed')} total={get_path(personality_drift, ['summary', 'total'], None)} score={get_path(personality_drift, ['summary', 'average_score'], None)}",
        ),
        check(
            "personality_runtime_audit_green",
            personality_runtime.get("trigger_state") == "GREEN",
            "blocker",
            f"state={personality_runtime.get('trigger_state')} selected_cards={get_path(personality_runtime, ['summary', 'selected_cards'], None)} drift_score={get_path(personality_runtime, ['summary', 'drift_average_score'], None)}",
        ),
        check(
            "virtual_context_memory_green",
            vcm_probe.get("trigger_state") == "GREEN"
            and get_path(vcm_status, ["summary", "vcm_bench_state"], "") == "GREEN",
            "blocker",
            f"probe={vcm_probe.get('trigger_state')} bench={get_path(vcm_status, ['summary', 'vcm_bench_state'], None)} pages={get_path(vcm_probe, ['summary', 'semantic_pages'], None)} faults={get_path(vcm_status, ['summary', 'fault_count'], None)}",
        ),
        check(
            "virtual_context_memory_training_admission_green",
            vcm_training.get("trigger_state") == "GREEN"
            and int(get_path(vcm_training, ["summary", "public_training_leaks"], 0) or 0) == 0
            and int(get_path(vcm_training, ["summary", "teacher_boundary_leaks"], 0) or 0) == 0,
            "blocker",
            f"state={vcm_training.get('trigger_state')} admitted={get_path(vcm_training, ['summary', 'admitted_rows'], None)} public_leaks={get_path(vcm_training, ['summary', 'public_training_leaks'], None)} teacher_leaks={get_path(vcm_training, ['summary', 'teacher_boundary_leaks'], None)}",
        ),
        check(
            "virtual_context_memory_consumer_audit_green",
            vcm_consumer_audit.get("trigger_state") == "GREEN"
            and int(get_path(vcm_consumer_audit, ["summary", "high_value_vcm_count"], 0) or 0)
            == int(get_path(vcm_consumer_audit, ["summary", "high_value_consumer_count"], 0) or 0),
            "blocker",
            f"state={vcm_consumer_audit.get('trigger_state')} high_value={get_path(vcm_consumer_audit, ['summary', 'high_value_vcm_count'], None)}/{get_path(vcm_consumer_audit, ['summary', 'high_value_consumer_count'], None)} packet_only={get_path(vcm_consumer_audit, ['summary', 'packet_only_consumer_count'], None)}",
        ),
        check(
            "virtual_context_memory_task_context_bridge_green",
            vcm_task_context_bridge.get("trigger_state") == "GREEN"
            and int(get_path(vcm_task_context_bridge, ["summary", "high_priority_ready_count"], 0) or 0)
            == int(get_path(vcm_task_context_bridge, ["summary", "high_priority_task_family_count"], 0) or 0)
            and int(get_path(vcm_task_context_bridge, ["summary", "public_training_rows_written"], 0) or 0) == 0
            and int(get_path(vcm_task_context_bridge, ["summary", "external_inference_calls"], 0) or 0) == 0
            and int(get_path(vcm_task_context_bridge, ["summary", "fallback_return_count"], 0) or 0) == 0,
            "blocker",
            f"state={vcm_task_context_bridge.get('trigger_state')} high_priority={get_path(vcm_task_context_bridge, ['summary', 'high_priority_ready_count'], None)}/{get_path(vcm_task_context_bridge, ['summary', 'high_priority_task_family_count'], None)} task_families={get_path(vcm_task_context_bridge, ['summary', 'ready_task_family_count'], None)}/{get_path(vcm_task_context_bridge, ['summary', 'task_family_count'], None)} contexts={get_path(vcm_task_contexts, ['task_context_count'], None)}",
        ),
        check(
            "virtual_context_memory_context_recovery_green",
            vcm_context_recovery.get("trigger_state") == "GREEN"
            and float(get_path(vcm_context_recovery, ["summary", "vcm_answer_accuracy"], 0.0) or 0.0)
            > float(get_path(vcm_context_recovery, ["summary", "best_baseline_answer_accuracy"], 0.0) or 0.0)
            and int(vcm_context_recovery.get("external_inference_calls") or 0) == 0
            and int(vcm_context_recovery.get("public_training_rows_written") or 0) == 0
            and int(vcm_context_recovery.get("fallback_return_count") or 0) == 0,
            "blocker",
            (
                f"state={vcm_context_recovery.get('trigger_state')} "
                f"vcm_accuracy={get_path(vcm_context_recovery, ['summary', 'vcm_answer_accuracy'], None)} "
                f"best_baseline={get_path(vcm_context_recovery, ['summary', 'best_baseline_answer_accuracy'], None)} "
                f"fallbacks={vcm_context_recovery.get('fallback_return_count')}"
            ),
        ),
        check(
            "virtual_context_memory_on_off_ablation_green",
            vcm_on_off_ablation.get("trigger_state") == "GREEN"
            and float(get_path(vcm_on_off_ablation, ["summary", "answer_accuracy_lift"], 0.0) or 0.0) > 0.0
            and int(get_path(vcm_on_off_ablation, ["summary", "win_counts", "off_only"], 1) or 0) == 0
            and int(vcm_on_off_ablation.get("fallback_return_count") or 0) == 0
            and int(vcm_on_off_ablation.get("external_inference_calls") or 0) == 0
            and int(vcm_on_off_ablation.get("public_training_rows_written") or 0) == 0,
            "blocker",
            (
                f"state={vcm_on_off_ablation.get('trigger_state')} "
                f"on={get_path(vcm_on_off_ablation, ['summary', 'vcm_on_answer_accuracy'], None)} "
                f"off={get_path(vcm_on_off_ablation, ['summary', 'vcm_off_answer_accuracy'], None)} "
                f"lift={get_path(vcm_on_off_ablation, ['summary', 'answer_accuracy_lift'], None)} "
                f"off_only={get_path(vcm_on_off_ablation, ['summary', 'win_counts', 'off_only'], None)}"
            ),
        ),
        check(
            "virtual_context_memory_public_memory_calibration_clean",
            vcm_public_memory_calibration.get("policy") == "project_theseus_vcm_public_memory_calibration_v1"
            and vcm_public_memory_calibration.get("trigger_state") in {"GREEN", "YELLOW"}
            and vcm_public_memory_calibration.get("calibration_mode") == "metadata_clean_public_benchmark_card_slice"
            and int(get_path(vcm_public_memory_calibration, ["summary", "external_inference_calls"], 1) or 0) == 0
            and int(get_path(vcm_public_memory_calibration, ["summary", "public_training_rows_written"], 1) or 0) == 0
            and int(get_path(vcm_public_memory_calibration, ["summary", "fallback_return_count"], 1) or 0) == 0
            and get_path(vcm_public_memory_calibration, ["summary", "official_payload_item_score_claimed"], True) is False
            and public_payload_counters_zero(vcm_public_memory_calibration),
            "blocker",
            (
                f"state={vcm_public_memory_calibration.get('trigger_state')} "
                f"mode={vcm_public_memory_calibration.get('calibration_mode')} "
                f"benchmarks={get_path(vcm_public_memory_calibration, ['summary', 'benchmark_count'], None)} "
                f"payload_score_claimed={get_path(vcm_public_memory_calibration, ['summary', 'official_payload_item_score_claimed'], None)}"
            ),
        ),
        check(
            "virtual_context_memory_public_memory_readiness_clean",
            public_memory_readiness_clean(vcm_public_memory_readiness),
            "blocker",
            (
                f"state={vcm_public_memory_readiness.get('trigger_state')} "
                f"allowed={vcm_public_memory_readiness.get('public_calibration_allowed')} "
                f"slice={vcm_public_memory_readiness.get('recommended_public_slice_id')} "
                f"hard_failures={len(list_value(vcm_public_memory_readiness.get('hard_failures')))}"
            ),
        ),
        check(
            "virtual_context_memory_public_memory_prompt_calibration_clean",
            public_memory_prompt_calibration_clean(vcm_public_memory_prompt_calibration),
            "blocker",
            (
                f"state={vcm_public_memory_prompt_calibration.get('trigger_state')} "
                f"mode={vcm_public_memory_prompt_calibration.get('calibration_mode')} "
                f"scored={get_path(vcm_public_memory_prompt_calibration, ['summary', 'scored_item_count'], None)} "
                f"benchmarks={sorted(dict_value(get_path(vcm_public_memory_prompt_calibration, ['summary', 'per_benchmark'], {})).keys())} "
                f"quarantined={get_path(vcm_public_memory_prompt_calibration, ['public_boundary', 'public_payloads_quarantined'], None)} "
                f"fallbacks={get_path(vcm_public_memory_prompt_calibration, ['summary', 'fallback_return_count'], None)}"
            ),
        ),
        check(
            "virtual_context_memory_public_memory_prompt_no_regression",
            public_memory_prompt_positive_transfer(vcm_public_memory_prompt_calibration),
            "blocker",
            (
                f"scored={get_path(vcm_public_memory_prompt_calibration, ['summary', 'scored_item_count'], None)} "
                f"vcm_on={get_path(vcm_public_memory_prompt_calibration, ['summary', 'vcm_on_pass_rate'], None)} "
                f"vcm_off={get_path(vcm_public_memory_prompt_calibration, ['summary', 'vcm_off_pass_rate'], None)} "
                f"delta={get_path(vcm_public_memory_prompt_calibration, ['summary', 'vcm_over_flat_tail_delta'], None)} "
                f"off_only={get_path(vcm_public_memory_prompt_calibration, ['summary', 'win_counts', 'vcm_off'], None)}"
            ),
        ),
        check(
            "virtual_context_memory_public_memory_private_repair_clean",
            public_memory_private_repair_clean(vcm_public_memory_private_repair),
            "blocker",
            (
                f"state={vcm_public_memory_private_repair.get('trigger_state')} "
                f"private_only={vcm_public_memory_private_repair.get('private_only')} "
                f"fixtures={vcm_public_memory_private_repair.get('fixture_count')} "
                f"categories={vcm_public_memory_private_repair.get('residual_categories')}"
            ),
        ),
        check(
            "virtual_context_memory_longmemeval_private_residual_green",
            longmemeval_private_residual_clean(vcm_longmemeval_private_residual),
            "blocker",
            (
                f"state={vcm_longmemeval_private_residual.get('trigger_state')} "
                f"cases={get_path(vcm_longmemeval_private_residual, ['summary', 'case_count'], None)} "
                f"vcm={get_path(vcm_longmemeval_private_residual, ['summary', 'vcm_on_pass_rate'], None)} "
                f"min_type={get_path(vcm_longmemeval_private_residual, ['summary', 'minimum_major_question_type_pass_rate'], None)} "
                f"delta={get_path(vcm_longmemeval_private_residual, ['summary', 'vcm_over_best_single_non_vcm_delta'], None)} "
                f"proposal={get_path(vcm_longmemeval_private_residual, ['future_public_calibration_proposal', 'proposal_state'], None)}"
            ),
        ),
        check(
            "virtual_context_memory_evidence_gauntlet_green",
            vcm_evidence_gauntlet_clean(vcm_evidence_gauntlet),
            "blocker",
            (
                f"state={vcm_evidence_gauntlet.get('trigger_state')} "
                f"cases={get_path(vcm_evidence_gauntlet, ['summary', 'case_count'], None)} "
                f"vcm={get_path(vcm_evidence_gauntlet, ['summary', 'vcm_on_pass_rate'], None)} "
                f"best_non_vcm={get_path(vcm_evidence_gauntlet, ['summary', 'best_single_non_vcm_pass_rate'], None)} "
                f"delta={get_path(vcm_evidence_gauntlet, ['summary', 'vcm_over_best_single_non_vcm_delta'], None)} "
                f"min_family={get_path(vcm_evidence_gauntlet, ['summary', 'minimum_major_family_pass_rate'], None)} "
                f"abstention={get_path(vcm_evidence_gauntlet, ['summary', 'abstention'], {})}"
            ),
        ),
        check(
            "virtual_context_memory_hard_memory_private_green",
            vcm_hard_memory_private_clean(vcm_hard_memory_private),
            "blocker",
            (
                f"state={vcm_hard_memory_private.get('trigger_state')} "
                f"cases={get_path(vcm_hard_memory_private, ['summary', 'case_count'], None)} "
                f"families={get_path(vcm_hard_memory_private, ['summary', 'family_count'], None)} "
                f"buckets={get_path(vcm_hard_memory_private, ['summary', 'length_bucket_count'], None)} "
                f"vcm={get_path(vcm_hard_memory_private, ['summary', 'vcm_on_pass_rate'], None)} "
                f"best_non_vcm={get_path(vcm_hard_memory_private, ['summary', 'best_single_non_vcm_pass_rate'], None)} "
                f"delta={get_path(vcm_hard_memory_private, ['summary', 'vcm_over_best_single_non_vcm_delta'], None)} "
                f"min_family={get_path(vcm_hard_memory_private, ['summary', 'minimum_family_pass_rate'], None)}"
            ),
        ),
        check(
            "virtual_context_memory_hard_public_readiness_recorded",
            vcm_hard_memory_readiness_recorded(vcm_hard_memory_readiness),
            "warning",
            (
                f"state={vcm_hard_memory_readiness.get('trigger_state')} "
                f"public_rows={get_path(vcm_hard_memory_readiness, ['summary', 'current_public_prompt_rows_scored'], None)} "
                f"target={get_path(vcm_hard_memory_readiness, ['summary', 'public_row_target'], None)} "
                f"metadata_ready={get_path(vcm_hard_memory_readiness, ['summary', 'metadata_ready_count'], None)} "
                f"blocked_or_queued={get_path(vcm_hard_memory_readiness, ['summary', 'blocked_or_queued_count'], None)}"
            ),
        ),
        check(
            "virtual_context_memory_prefetch_regret_green",
            vcm_prefetch_regret.get("trigger_state") == "GREEN"
            and int(get_path(vcm_prefetch_regret, ["summary", "external_inference_calls"], 1) or 0) == 0
            and int(get_path(vcm_prefetch_regret, ["summary", "public_training_rows_written"], 1) or 0) == 0
            and int(get_path(vcm_prefetch_regret, ["summary", "fallback_return_count"], 1) or 0) == 0
            and get_path(vcm_prefetch_regret, ["summary", "usage_events_private"], False) is True,
            "blocker",
            (
                f"state={vcm_prefetch_regret.get('trigger_state')} "
                f"precision={get_path(vcm_prefetch_regret, ['summary', 'prefetch_precision'], None)} "
                f"miss_rate={get_path(vcm_prefetch_regret, ['summary', 'prefetch_miss_rate'], None)} "
                f"regret={get_path(vcm_prefetch_regret, ['summary', 'prefetch_regret'], None)}"
            ),
        ),
        check(
            "virtual_context_memory_runtime_claim_readiness_green",
            vcm_runtime_claim_readiness.get("trigger_state") == "GREEN"
            and get_path(vcm_runtime_claim_readiness, ["summary", "runtime_profile_claimed"], True) is False
            and get_path(vcm_runtime_claim_readiness, ["summary", "native_kv_cache_claimed"], True) is False
            and float(get_path(vcm_runtime_claim_readiness, ["summary", "cache_key_complete_rate"], 0.0) or 0.0) >= 1.0
            and int(get_path(vcm_runtime_claim_readiness, ["summary", "rejected_semantic_claims"], 1) or 0) == 0
            and int(get_path(vcm_runtime_claim_readiness, ["summary", "fallback_return_count"], 1) or 0) == 0,
            "blocker",
            (
                f"state={vcm_runtime_claim_readiness.get('trigger_state')} "
                f"runtime_claimed={get_path(vcm_runtime_claim_readiness, ['summary', 'runtime_profile_claimed'], None)} "
                f"key_rate={get_path(vcm_runtime_claim_readiness, ['summary', 'cache_key_complete_rate'], None)} "
                f"accepted={get_path(vcm_runtime_claim_readiness, ['summary', 'accepted_semantic_claims'], None)} "
                f"rejected={get_path(vcm_runtime_claim_readiness, ['summary', 'rejected_semantic_claims'], None)}"
            ),
        ),
        check(
            "virtual_context_memory_release_conformance_core_ready",
            bool(get_path(vcm_release_conformance, ["summary", "core_profiles_ready"], False))
            and int(get_path(vcm_release_conformance, ["summary", "hard_failure_count"], -1)) == 0
            and vcm_release_runtime_claim_ok,
            "blocker",
            (
                f"state={vcm_release_conformance.get('trigger_state')} "
                f"profiles={get_path(vcm_release_conformance, ['summary', 'profile_states'], {})} "
                f"runtime_claimed={vcm_runtime_profile_claimed} "
                f"native_claimable={get_path(vcm_release_conformance, ['summary', 'native_runtime_claimable'], None)} "
                f"native_lifecycle={get_path(vcm_release_conformance, ['summary', 'native_prefix_kv_lifecycle_test_passed'], None)} "
                f"native_blockers={get_path(vcm_release_conformance, ['summary', 'native_runtime_blocker_count'], None)}"
            ),
        ),
        check(
            "belief_update_governance_ready",
            belief_governance.get("status") in {"ready", "evaluated"},
            "blocker",
            f"status={belief_governance.get('status')} ledger_entries={get_path(belief_governance, ['summary', 'ledger_entries'], None)}",
        ),
        check(
            "no_quarantined_belief_updates",
            int(get_path(belief_governance, ["summary", "quarantined"], 0) or 0) == 0,
            "blocker",
            f"quarantined={get_path(belief_governance, ['summary', 'quarantined'], None)} needs_review={get_path(belief_governance, ['summary', 'needs_review'], None)}",
        ),
        check(
            "legacy_runtime_governance_ready",
            legacy_runtime.get("trigger_state") in {"GREEN", "YELLOW"}
            and bool(legacy_runtime.get("ready_for_teacher_work")),
            "blocker",
            f"state={legacy_runtime.get('trigger_state')} teacher={legacy_runtime.get('ready_for_teacher_work')} warnings={get_path(legacy_runtime, ['summary', 'warning_count'], None)} failed={get_path(legacy_runtime, ['summary', 'failed_gates'], [])}",
        ),
        check(
            "legacy_runtime_candidate_promotion_clean",
            bool(legacy_runtime.get("ready_for_candidate_promotion")),
            "warning",
            f"state={legacy_runtime.get('trigger_state')} warnings={get_path(legacy_runtime, ['summary', 'warning_count'], None)} proxy={get_path(legacy_runtime, ['summary', 'proxy_truth_state'], None)} coherence={get_path(legacy_runtime, ['summary', 'coherence_trigger_state'], None)}",
        ),
        check(
            "legacy_port_runtime_enforcement_bounded_ready",
            bool(legacy_runtime_enforcement.get("ready_for_bounded_autonomy")),
            "blocker",
            f"state={get_path(legacy_runtime_enforcement, ['summary', 'trigger_state'], None)} blockers={legacy_runtime_enforcement.get('blockers', [])}",
        ),
        check(
            "legacy_port_runtime_enforcement_long_run_ready",
            bool(legacy_runtime_enforcement.get("ready_for_long_autonomy")),
            "warning",
            f"state={get_path(legacy_runtime_enforcement, ['summary', 'trigger_state'], None)} blockers={legacy_runtime_enforcement.get('blockers', [])}",
        ),
        check(
            "legacy_port_runtime_enforcement_candidate_ready",
            bool(legacy_runtime_enforcement.get("ready_for_candidate_promotion")),
            "warning",
            f"state={get_path(legacy_runtime_enforcement, ['summary', 'trigger_state'], None)} blockers={legacy_runtime_enforcement.get('blockers', [])}",
        ),
        check(
            "coherence_delirium_allows_long_autonomy",
            bool(coherence_gate.get("allows_long_autonomy")),
            "blocker",
            f"state={coherence_gate.get('trigger_state')} source={coherence_gate.get('source_trigger_state')} coherence={coherence_gate.get('coherence_score')} delirium={coherence_gate.get('delirium_score')} blockers={coherence_gate.get('blockers', [])}",
        ),
        check(
            "coherence_delirium_candidate_promotion_clean",
            bool(coherence_gate.get("allows_candidate_promotion")),
            "warning",
            f"state={coherence_gate.get('trigger_state')} delirium={coherence_gate.get('delirium_score')} candidate_blockers={coherence_gate.get('candidate_blockers', [])}",
        ),
        check(
            "legacy_training_source_admission_ready",
            legacy_training_sources.get("trigger_state") in {"GREEN", "YELLOW"}
            and int(get_path(legacy_training_sources, ["summary", "serious_training_ready"], 0) or 0) > 0
            and int(get_path(legacy_training_sources, ["summary", "hash_mismatches"], 0) or 0) == 0,
            "warning",
            f"state={legacy_training_sources.get('trigger_state')} serious={get_path(legacy_training_sources, ['summary', 'serious_training_ready'], None)} hash_mismatches={get_path(legacy_training_sources, ['summary', 'hash_mismatches'], None)}",
        ),
        check(
            "legacy_rl_environment_admission_ready",
            legacy_rl_envs.get("trigger_state") in {"GREEN", "YELLOW"}
            and int(get_path(legacy_rl_envs, ["summary", "p0_smoke_lane"], 0) or 0) > 0
            and int(get_path(legacy_rl_envs, ["summary", "hardware_gated_envs"], 0) or 0) >= 0,
            "warning",
            f"state={legacy_rl_envs.get('trigger_state')} envs={get_path(legacy_rl_envs, ['summary', 'environments'], None)} p0={get_path(legacy_rl_envs, ['summary', 'p0_smoke_lane'], None)} hardware_gated={get_path(legacy_rl_envs, ['summary', 'hardware_gated_envs'], None)}",
        ),
        check(
            "legacy_training_tiny_sample_ready",
            legacy_training_sample.get("trigger_state") in {"GREEN", "YELLOW"}
            and int(get_path(legacy_training_sample, ["summary", "sample_rows"], 0) or 0) > 0,
            "warning",
            f"state={legacy_training_sample.get('trigger_state')} rows={get_path(legacy_training_sample, ['summary', 'sample_rows'], None)} lanes={get_path(legacy_training_sample, ['summary', 'lane_counts'], {})}",
        ),
        check(
            "legacy_rl_smoke_plan_ready",
            legacy_rl_smoke.get("trigger_state") in {"GREEN", "YELLOW"}
            and int(get_path(legacy_rl_smoke, ["summary", "planned_envs"], 0) or 0) > 0,
            "warning",
            f"state={legacy_rl_smoke.get('trigger_state')} planned={get_path(legacy_rl_smoke, ['summary', 'planned_envs'], None)} ready={get_path(legacy_rl_smoke, ['summary', 'ready_for_seeded_smoke'], None)} pending={get_path(legacy_rl_smoke, ['summary', 'pending_dependency'], None)} source_present_pending_install={get_path(legacy_rl_smoke, ['summary', 'source_present_pending_install'], None)} runner_pending_adapter={get_path(legacy_rl_smoke, ['summary', 'runner_pending_adapter'], None)}",
        ),
        check(
            "trace_fabric_capsule_admission_ready",
            trace_capsules.get("trigger_state") in {"GREEN", "YELLOW"}
            and int(get_path(trace_capsules, ["summary", "accepted_metadata_only"], 0) or 0) > 0
            and int(get_path(trace_capsules, ["summary", "raw_payload_key_hits"], 0) or 0) == 0,
            "warning",
            f"state={trace_capsules.get('trigger_state')} accepted={get_path(trace_capsules, ['summary', 'accepted_metadata_only'], None)} raw_payload_keys={get_path(trace_capsules, ['summary', 'raw_payload_key_hits'], None)}",
        ),
        check(
            "trace_fabric_capsule_materialization_ready",
            trace_materialization.get("trigger_state") in {"GREEN", "YELLOW"}
            and int(get_path(trace_materialization, ["summary", "materialized_rows"], 0) or 0) > 0
            and int(get_path(trace_materialization, ["summary", "raw_payload_rows"], 0) or 0) == 0,
            "warning",
            f"state={trace_materialization.get('trigger_state')} rows={get_path(trace_materialization, ['summary', 'materialized_rows'], None)} raw={get_path(trace_materialization, ['summary', 'raw_payload_rows'], None)} rejections={get_path(trace_materialization, ['summary', 'rejections'], {})}",
        ),
        check(
            "legacy_adapter_bank_training_plan_ready",
            legacy_adapter_plan.get("trigger_state") in {"GREEN", "YELLOW"}
            and bool(legacy_adapter_plan.get("ready_for_zero_param_dry_run"))
            and int(get_path(legacy_adapter_plan, ["summary", "plan_rows"], 0) or 0) > 0
            and int(get_path(legacy_adapter_plan, ["summary", "external_inference_calls"], 0) or 0) == 0,
            "warning",
            f"state={legacy_adapter_plan.get('trigger_state')} rows={get_path(legacy_adapter_plan, ['summary', 'plan_rows'], None)} selected={get_path(legacy_adapter_plan, ['summary', 'selected_adapters'], [])} zero_param={get_path(legacy_adapter_plan, ['summary', 'zero_param_lanes'], [])}",
        ),
        check(
            "legacy_active_inference_pilot_ready",
            active_inference_pilot.get("trigger_state") in {"GREEN", "YELLOW"}
            and bool(active_inference_pilot.get("ready_for_world_model_training_signal"))
            and int(get_path(active_inference_pilot, ["summary", "action_rankings"], 0) or 0) > 0
            and int(get_path(active_inference_pilot, ["summary", "accepted_belief_updates"], 0) or 0) > 0
            and int(get_path(active_inference_pilot, ["summary", "quarantined_belief_updates"], 0) or 0) == 0
            and int(get_path(active_inference_pilot, ["summary", "external_inference_calls"], 0) or 0) == 0,
            "warning",
            f"state={active_inference_pilot.get('trigger_state')} error={get_path(active_inference_pilot, ['summary', 'mean_prediction_error'], None)} rankings={get_path(active_inference_pilot, ['summary', 'action_rankings'], None)} updates={get_path(active_inference_pilot, ['summary', 'accepted_belief_updates'], None)}",
        ),
        check(
            "whitecell_threat_memory_available",
            whitecell.get("policy") == "beastbrain_whitecell_local_threat_memory_v0"
            and bool(whitecell.get("local_only")),
            "warning",
            f"state={whitecell.get('trigger_state')} local_only={whitecell.get('local_only')}",
        ),
        check(
            "whitecell_no_active_block_and_escalate",
            not whitecell_blockers,
            "warning",
            f"active_blockers={whitecell_blockers}",
        ),
        check(
            "teacher_policy_valid",
            bool(teacher_policy.get("codex_command")) and teacher_policy.get("default_mode") == "proposal",
            "blocker",
            f"provider={teacher_policy.get('provider')} model={teacher_policy.get('model')} mode={teacher_policy.get('default_mode')}",
        ),
        check(
            "teacher_architecture_proposal_mode_available",
            teacher_architecture_proposal_mode_available(teacher_policy, autonomy_policy),
            "blocker",
            (
                f"default_mode={teacher_policy.get('default_mode')} "
                f"proposal_sandbox={teacher_policy.get('proposal_sandbox')} "
                f"allowed_reasons={teacher_policy.get('allowed_reasons')} "
                f"proposal_only_no_distillation={get_path(teacher_policy, ['budget', 'proposal_only_no_distillation'], None)}"
            ),
        ),
        check(
            "teacher_cli_available",
            teacher_cli_available,
            "blocker" if args.require_teacher_cli else "warning",
            f"command={teacher_policy.get('codex_command') or 'codex'} resolved={teacher_cli_path}",
        ),
        check(
            "teacher_wrapper_smoke_completed",
            teacher_smoke_passed,
            "blocker" if args.require_teacher_cli else "warning",
            f"latest_status={teacher_smoke.get('status')} latest_returncode={teacher_smoke.get('returncode')} completed_log_calls={completed_teacher_calls(teacher_calls)}",
        ),
        check(
            "daemon_scripts_present",
            all((ROOT / path).exists() for path in [
                "scripts/sparkstream_daemon.py",
                "scripts/autonomy_cycle.py",
                "scripts/start_sparkstream.ps1",
            ]),
            "blocker",
            "dashboard/daemon/autonomy entrypoints exist",
        ),
        check(
            "dashboard_present",
            all((ROOT / path).exists() for path in [
                "scripts/sparkstream_dashboard.py",
                "dashboard/index.html",
                "dashboard/app.js",
                "dashboard/styles.css",
            ]),
            "blocker",
            "local dashboard assets exist",
        ),
        check(
            "hive_policy_available",
            hive_policy.get("policy") == "project_theseus_hive_policy_v0",
            "warning",
            f"enabled={hive_policy.get('enabled')} remote_secret_required={get_path(hive_policy, ['security', 'requires_shared_secret_for_remote_tasks'], None)}",
        ),
        check(
            "hive_node_probe_available",
            hive_status.get("policy") == "project_theseus_hive_node_status_v0",
            "warning",
            f"node={hive_status.get('node_name')} capabilities={len(hive_status.get('capabilities') or [])}",
        ),
        check(
            "hive_scheduler_available",
            hive_scheduler.get("policy") == "project_theseus_hive_scheduler_v0",
            "warning",
            f"nodes={get_path(hive_scheduler, ['summary', 'nodes'], None)} peers={hive_peers.get('peer_count')}",
        ),
        check(
            "license_registration_complete",
            bool(license_status.get("registration_complete")),
            "blocker",
            f"tier={get_path(license_status, ['entitlement', 'tier'], None)} source={get_path(license_status, ['entitlement', 'source'], None)}",
        ),
        check(
            "license_allows_distributed_worker_chunks",
            bool(license_worker_chunks.get("allowed")),
            "blocker",
            f"next_action={license_worker_chunks.get('next_action')}",
        ),
        check(
            "checkpoint_registry_available",
            bool((checkpoint_registry.get("checkpoints") or [])),
            "warning",
            f"checkpoints={len(checkpoint_registry.get('checkpoints') or [])}",
        ),
        check(
            "accepted_candidate_backup_policy_available",
            checkpoint_backup_policy.get("policy") == "project_theseus_checkpoint_backup_policy_v0",
            "warning",
            f"github_enabled={get_path(checkpoint_backup_policy, ['providers', 'github', 'enabled'], None)} google_drive_enabled={get_path(checkpoint_backup_policy, ['providers', 'google_drive', 'enabled'], None)}",
        ),
        check(
            "accepted_candidate_backup_last_report_available",
            checkpoint_backup.get("policy") == "project_theseus_checkpoint_backup_report_v0",
            "warning",
            f"status={checkpoint_backup.get('status')} checkpoint={checkpoint_backup.get('checkpoint_id')}",
        ),
        check(
            "real_routing_traces_available",
            (reports / "routing_memory_real_traces.jsonl").exists()
            and (reports / "routing_memory_real_traces.jsonl").stat().st_size > 0,
            "warning",
            "real traces feed router and arm lifecycle updates",
        ),
        check(
            "stop_flag_clear",
            not (reports / "sparkstream_stop.flag").exists(),
            "warning",
            "daemon clears stop flag on start, but a clear workspace is easier to reason about",
        ),
        check(
            "pause_flag_clear",
            not (reports / "sparkstream_pause.flag").exists(),
            "warning",
            "daemon clears pause flag on start, but a clear workspace is easier to reason about",
        ),
        check(
            "policy_profile_allowed",
            args.profile in set(autonomy_policy.get("allowed_profiles") or []),
            "blocker",
            f"profile={args.profile} allowed={autonomy_policy.get('allowed_profiles')}",
        ),
        check(
            "long_run_guardrails_enabled",
            long_run_guardrails_enabled(autonomy_policy),
            "blocker",
            "teacher apply is guarded by branch-and-gate self-evolution policy; autonomous network use is license-gated and blocks uncertain/bulk/commercial sources",
        ),
    ]

    blocker_failures = [row for row in checks if row["severity"] == "blocker" and not row["passed"]]
    warning_failures = [row for row in checks if row["severity"] == "warning" and not row["passed"]]
    candidate_blockers = failed_candidate_gates(candidate)
    ready_for_autonomous_training = not blocker_failures
    local_macos_smoke_exempt_gates = {
        "preflight_allows_heavy_training",
        "seed55_frontier_artifact_exists",
        # A Mac-local smoke proves one bounded local MLX/CPU worker chunk. It
        # must not be promoted into full legacy-port runtime readiness or model
        # promotion evidence.
        "legacy_runtime_governance_ready",
        "legacy_port_runtime_enforcement_bounded_ready",
    }
    local_macos_smoke_blockers = [
        row
        for row in blocker_failures
        if row["gate"] not in local_macos_smoke_exempt_gates
    ]
    ready_for_local_macos_smoke_training = (
        local_macos_smoke_ready and not local_macos_smoke_blockers
    )
    ready_for_candidate_promotion = (
        ready_for_autonomous_training
        and bool(candidate.get("promote"))
        and bool(coherence_gate.get("allows_candidate_promotion"))
    )
    ready_for_teacher_enabled_run = (ready_for_autonomous_training or ready_for_local_macos_smoke_training) and (
        teacher_cli_available
        and teacher_smoke_passed
        or not args.require_teacher_cli
    )
    trigger_state = (
        "GREEN"
        if ready_for_autonomous_training
        else "YELLOW"
        if ready_for_local_macos_smoke_training
        else "RED"
    )

    report = {
        "policy": "sparkstream_autonomy_launch_readiness_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "profile": args.profile,
        "summary": {
            "profile": args.profile,
            "ready_for_autonomous_training": ready_for_autonomous_training,
            "ready_for_local_macos_smoke_training": ready_for_local_macos_smoke_training,
            "ready_for_teacher_enabled_run": ready_for_teacher_enabled_run,
            "ready_for_candidate_promotion": ready_for_candidate_promotion,
            "blocker_failure_count": len(blocker_failures),
            "warning_failure_count": len(warning_failures),
            "candidate_blocker_count": len(candidate_blockers),
            "blocker_failures": [row.get("gate") for row in blocker_failures],
            "warning_failures": [row.get("gate") for row in warning_failures],
            "resource_governor_state": resource.get("trigger_state"),
            "resource_can_run": get_path(resource, ["decision", "can_run_requested_profile"], None),
            "resource_execution_owner": get_path(resource, ["decision", "execution_owner"], None),
            "vcm_release_conformance_state": vcm_release_conformance.get("trigger_state"),
            "vcm_runtime_profile_claimed": vcm_runtime_profile_claimed,
            "vcm_native_runtime_claimable": get_path(vcm_release_conformance, ["summary", "native_runtime_claimable"], None),
            "macos_preflight_state": macos_preflight.get("state"),
            "teacher_proposal_mode_available": teacher_architecture_proposal_mode_available(teacher_policy, autonomy_policy),
            "teacher_cli_available": teacher_cli_available,
            "teacher_smoke_passed": teacher_smoke_passed,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "public_training_rows_written": 0,
        },
        "ready_for_autonomous_training": ready_for_autonomous_training,
        "ready_for_local_macos_smoke_training": ready_for_local_macos_smoke_training,
        "ready_for_teacher_enabled_run": ready_for_teacher_enabled_run,
        "ready_for_candidate_promotion": ready_for_candidate_promotion,
        "local_macos_smoke_exempt_gates": sorted(local_macos_smoke_exempt_gates),
        "local_macos_smoke_blockers": local_macos_smoke_blockers,
        "blocker_failures": blocker_failures,
        "warning_failures": warning_failures,
        "candidate_blockers": candidate_blockers,
        "attd": {
            "trigger_state": attd.get("trigger_state"),
            "attd_score": attd.get("attd_score"),
            "maintenance_packets": attd_packets.get("packet_count"),
        },
        "autoresearch_gap_audit": {
            "trigger_state": (autoresearch_audit.get("summary") or {}).get("trigger_state"),
            "gap_count": (autoresearch_audit.get("summary") or {}).get("gap_count"),
            "ledger_entries": (autoresearch_audit.get("summary") or {}).get("ledger_entries"),
            "needs_baseline": (autoresearch_audit.get("summary") or {}).get("needs_baseline"),
        },
        "personality_context": {
            "status": personality_context.get("status"),
            "selected_cards": get_path(personality_context, ["summary", "selected_cards"], None),
            "hard_safety_invariants": get_path(personality_context, ["summary", "hard_safety_invariants"], None),
            "anti_drift_rules": get_path(personality_context, ["summary", "anti_drift_rules"], None),
        },
        "personality_drift": {
            "passed": personality_drift.get("passed"),
            "average_score": get_path(personality_drift, ["summary", "average_score"], None),
            "failed_cases": get_path(personality_drift, ["summary", "failed_cases"], None),
        },
        "personality_runtime": {
            "trigger_state": personality_runtime.get("trigger_state"),
            "selected_cards": get_path(personality_runtime, ["summary", "selected_cards"], None),
            "drift_average_score": get_path(personality_runtime, ["summary", "drift_average_score"], None),
        },
        "virtual_context_memory": {
            "trigger_state": vcm_probe.get("trigger_state"),
            "bench_state": get_path(vcm_status, ["summary", "vcm_bench_state"], None),
            "training_admission_state": vcm_training.get("trigger_state"),
            "consumer_audit_state": vcm_consumer_audit.get("trigger_state"),
            "task_context_bridge_state": vcm_task_context_bridge.get("trigger_state"),
            "task_context_family_count": get_path(vcm_task_context_bridge, ["summary", "task_family_count"], None),
            "task_context_ready_family_count": get_path(vcm_task_context_bridge, ["summary", "ready_task_family_count"], None),
            "task_context_high_priority_ready": get_path(vcm_task_context_bridge, ["summary", "high_priority_ready_count"], None),
            "task_context_high_priority_total": get_path(vcm_task_context_bridge, ["summary", "high_priority_task_family_count"], None),
            "task_context_unique_selected_pages": get_path(vcm_task_context_bridge, ["summary", "unique_selected_page_count"], None),
            "context_recovery_state": vcm_context_recovery.get("trigger_state"),
            "on_off_ablation_state": vcm_on_off_ablation.get("trigger_state"),
            "on_off_answer_lift": get_path(vcm_on_off_ablation, ["summary", "answer_accuracy_lift"], None),
            "public_memory_calibration_state": vcm_public_memory_calibration.get("trigger_state"),
            "public_memory_calibration_mode": vcm_public_memory_calibration.get("calibration_mode"),
            "public_memory_official_payload_score_claimed": get_path(vcm_public_memory_calibration, ["summary", "official_payload_item_score_claimed"], None),
            "public_memory_readiness_state": vcm_public_memory_readiness.get("trigger_state"),
            "public_memory_readiness_allowed": vcm_public_memory_readiness.get("public_calibration_allowed"),
            "public_memory_readiness_recommended_slice": vcm_public_memory_readiness.get("recommended_public_slice_id"),
            "public_memory_prompt_calibration_state": vcm_public_memory_prompt_calibration.get("trigger_state"),
            "public_memory_prompt_vcm_on_pass_rate": get_path(vcm_public_memory_prompt_calibration, ["summary", "vcm_on_pass_rate"], None),
            "public_memory_prompt_vcm_off_pass_rate": get_path(vcm_public_memory_prompt_calibration, ["summary", "vcm_off_pass_rate"], None),
            "public_memory_prompt_vcm_over_flat_tail_delta": get_path(vcm_public_memory_prompt_calibration, ["summary", "vcm_over_flat_tail_delta"], None),
            "public_memory_prompt_vcm_over_best_non_vcm_delta": get_path(vcm_public_memory_prompt_calibration, ["summary", "vcm_over_best_non_vcm_delta"], None),
            "public_memory_prompt_best_non_vcm": get_path(vcm_public_memory_prompt_calibration, ["summary", "best_non_vcm_memory_system", "system"], None),
            "public_memory_prompt_off_only_wins": get_path(vcm_public_memory_prompt_calibration, ["summary", "win_counts", "vcm_off"], None),
            "public_memory_prompt_item_manifest": get_path(vcm_public_memory_prompt_calibration, ["quarantine", "item_manifest"], None),
            "public_memory_prompt_item_manifest_hash": get_path(vcm_public_memory_prompt_calibration, ["quarantine", "item_manifest_hash"], None),
            "public_memory_prompt_source_context_token_distribution": get_path(vcm_public_memory_prompt_calibration, ["summary", "source_context_token_distribution"], None),
            "public_memory_prompt_per_length_bucket": get_path(vcm_public_memory_prompt_calibration, ["summary", "per_length_bucket"], None),
            "public_memory_prompt_forbidden_overlap_counts": get_path(vcm_public_memory_prompt_calibration, ["summary", "forbidden_overlap_counts"], None),
            "public_memory_prompt_private_repair_state": vcm_public_memory_private_repair.get("trigger_state"),
            "public_memory_prompt_private_residual_categories": vcm_public_memory_private_repair.get("residual_categories"),
            "longmemeval_private_residual_state": vcm_longmemeval_private_residual.get("trigger_state"),
            "longmemeval_private_residual_vcm_pass_rate": get_path(vcm_longmemeval_private_residual, ["summary", "vcm_on_pass_rate"], None),
            "longmemeval_private_residual_min_type_pass_rate": get_path(vcm_longmemeval_private_residual, ["summary", "minimum_major_question_type_pass_rate"], None),
            "longmemeval_private_residual_vcm_over_best_non_vcm_delta": get_path(vcm_longmemeval_private_residual, ["summary", "vcm_over_best_single_non_vcm_delta"], None),
            "evidence_gauntlet_state": vcm_evidence_gauntlet.get("trigger_state"),
            "evidence_gauntlet_cases": get_path(vcm_evidence_gauntlet, ["summary", "case_count"], None),
            "evidence_gauntlet_vcm_pass_rate": get_path(vcm_evidence_gauntlet, ["summary", "vcm_on_pass_rate"], None),
            "evidence_gauntlet_best_single_non_vcm": get_path(vcm_evidence_gauntlet, ["summary", "best_single_non_vcm_pass_rate"], None),
            "evidence_gauntlet_delta": get_path(vcm_evidence_gauntlet, ["summary", "vcm_over_best_single_non_vcm_delta"], None),
            "evidence_gauntlet_min_family": get_path(vcm_evidence_gauntlet, ["summary", "minimum_major_family_pass_rate"], None),
            "evidence_gauntlet_abstention": get_path(vcm_evidence_gauntlet, ["summary", "abstention"], None),
            "hard_memory_private_state": vcm_hard_memory_private.get("trigger_state"),
            "hard_memory_private_cases": get_path(vcm_hard_memory_private, ["summary", "case_count"], None),
            "hard_memory_private_family_count": get_path(vcm_hard_memory_private, ["summary", "family_count"], None),
            "hard_memory_private_length_bucket_count": get_path(vcm_hard_memory_private, ["summary", "length_bucket_count"], None),
            "hard_memory_private_vcm_pass_rate": get_path(vcm_hard_memory_private, ["summary", "vcm_on_pass_rate"], None),
            "hard_memory_private_best_single_non_vcm": get_path(vcm_hard_memory_private, ["summary", "best_single_non_vcm_pass_rate"], None),
            "hard_memory_private_delta": get_path(vcm_hard_memory_private, ["summary", "vcm_over_best_single_non_vcm_delta"], None),
            "hard_memory_private_min_family": get_path(vcm_hard_memory_private, ["summary", "minimum_family_pass_rate"], None),
            "hard_memory_readiness_state": vcm_hard_memory_readiness.get("trigger_state"),
            "hard_memory_readiness_public_rows": get_path(vcm_hard_memory_readiness, ["summary", "current_public_prompt_rows_scored"], None),
            "hard_memory_readiness_public_target": get_path(vcm_hard_memory_readiness, ["summary", "public_row_target"], None),
            "hard_memory_readiness_metadata_ready_count": get_path(vcm_hard_memory_readiness, ["summary", "metadata_ready_count"], None),
            "hard_memory_readiness_blocked_or_queued_count": get_path(vcm_hard_memory_readiness, ["summary", "blocked_or_queued_count"], None),
            "prefetch_regret_state": vcm_prefetch_regret.get("trigger_state"),
            "prefetch_regret": get_path(vcm_prefetch_regret, ["summary", "prefetch_regret"], None),
            "runtime_claim_readiness_state": vcm_runtime_claim_readiness.get("trigger_state"),
            "runtime_cache_key_complete_rate": get_path(vcm_runtime_claim_readiness, ["summary", "cache_key_complete_rate"], None),
            "release_conformance_state": vcm_release_conformance.get("trigger_state"),
            "release_conformance_core_ready": get_path(vcm_release_conformance, ["summary", "core_profiles_ready"], None),
            "release_conformance_profiles": get_path(vcm_release_conformance, ["summary", "profile_states"], None),
            "context_recovery_vcm_accuracy": get_path(vcm_context_recovery, ["summary", "vcm_answer_accuracy"], None),
            "context_recovery_best_baseline_accuracy": get_path(vcm_context_recovery, ["summary", "best_baseline_answer_accuracy"], None),
            "page_count": get_path(vcm_probe, ["summary", "semantic_pages"], None),
            "event_count": get_path(vcm_probe, ["summary", "event_count"], None),
            "fault_count": get_path(vcm_status, ["summary", "fault_count"], None),
            "packet_only_consumer_count": get_path(vcm_consumer_audit, ["summary", "packet_only_consumer_count"], None),
        },
        "belief_governance": {
            "status": belief_governance.get("status"),
            "ledger_entries": get_path(belief_governance, ["summary", "ledger_entries"], None),
            "quarantined": get_path(belief_governance, ["summary", "quarantined"], None),
        },
        "macos_training_preflight": {
            "state": macos_preflight.get("state"),
            "bounded_smoke_allowed": macos_preflight.get("bounded_smoke_allowed"),
            "long_training_allowed": macos_preflight.get("long_training_allowed"),
            "hard_failures": get_path(macos_preflight, ["summary", "hard_failures"], None),
            "worker_canary": get_path(macos_preflight, ["execution", "kind"], None),
        },
        "legacy_runtime_governance": {
            "trigger_state": legacy_runtime.get("trigger_state"),
            "ready_for_teacher_work": legacy_runtime.get("ready_for_teacher_work"),
            "ready_for_candidate_promotion": legacy_runtime.get("ready_for_candidate_promotion"),
            "warning_count": get_path(legacy_runtime, ["summary", "warning_count"], None),
            "failed_gates": get_path(legacy_runtime, ["summary", "failed_gates"], []),
            "taskspell_lock_hash": get_path(legacy_runtime, ["summary", "taskspell_lock_hash"], None),
        },
        "legacy_port_runtime_enforcement": {
            "trigger_state": get_path(legacy_runtime_enforcement, ["summary", "trigger_state"], None),
            "ready_for_bounded_autonomy": legacy_runtime_enforcement.get("ready_for_bounded_autonomy"),
            "ready_for_long_autonomy": legacy_runtime_enforcement.get("ready_for_long_autonomy"),
            "ready_for_candidate_promotion": legacy_runtime_enforcement.get("ready_for_candidate_promotion"),
            "ready_for_self_evolution": legacy_runtime_enforcement.get("ready_for_self_evolution"),
            "blockers": legacy_runtime_enforcement.get("blockers", []),
            "effect_records": get_path(legacy_runtime_enforcement, ["summary", "effect_records"], None),
            "planforge_nodes": get_path(legacy_runtime_enforcement, ["summary", "planforge_nodes"], None),
        },
        "coherence_delirium_gate": {
            "trigger_state": coherence_gate.get("trigger_state"),
            "source_trigger_state": coherence_gate.get("source_trigger_state"),
            "coherence_score": coherence_gate.get("coherence_score"),
            "delirium_score": coherence_gate.get("delirium_score"),
            "allows_long_autonomy": coherence_gate.get("allows_long_autonomy"),
            "allows_candidate_promotion": coherence_gate.get("allows_candidate_promotion"),
            "allows_self_edit": coherence_gate.get("allows_self_edit"),
            "allows_capability_expansion": coherence_gate.get("allows_capability_expansion"),
            "blockers": coherence_gate.get("blockers", []),
            "candidate_blockers": coherence_gate.get("candidate_blockers", []),
        },
        "legacy_ports": {
            "training_sources": {
                "trigger_state": legacy_training_sources.get("trigger_state"),
                "ready_local_verified": get_path(legacy_training_sources, ["summary", "ready_local_verified"], None),
                "serious_training_ready": get_path(legacy_training_sources, ["summary", "serious_training_ready"], None),
                "hash_mismatches": get_path(legacy_training_sources, ["summary", "hash_mismatches"], None),
                "admission_plan": legacy_training_sources.get("admission_plan_path"),
            },
            "training_sample": {
                "trigger_state": legacy_training_sample.get("trigger_state"),
                "sample_rows": get_path(legacy_training_sample, ["summary", "sample_rows"], None),
                "lane_counts": get_path(legacy_training_sample, ["summary", "lane_counts"], {}),
                "sample_path": legacy_training_sample.get("sample_path"),
            },
            "rl_environments": {
                "trigger_state": legacy_rl_envs.get("trigger_state"),
                "environments": get_path(legacy_rl_envs, ["summary", "environments"], None),
                "p0_smoke_lane": get_path(legacy_rl_envs, ["summary", "p0_smoke_lane"], None),
                "hardware_gated_envs": get_path(legacy_rl_envs, ["summary", "hardware_gated_envs"], None),
            },
            "rl_smoke_plan": {
                "trigger_state": legacy_rl_smoke.get("trigger_state"),
                "planned_envs": get_path(legacy_rl_smoke, ["summary", "planned_envs"], None),
                "ready_for_seeded_smoke": get_path(legacy_rl_smoke, ["summary", "ready_for_seeded_smoke"], None),
                "pending_dependency": get_path(legacy_rl_smoke, ["summary", "pending_dependency"], None),
                "source_present_pending_install": get_path(legacy_rl_smoke, ["summary", "source_present_pending_install"], None),
                "runner_pending_adapter": get_path(legacy_rl_smoke, ["summary", "runner_pending_adapter"], None),
                "plan_path": legacy_rl_smoke.get("plan_path"),
            },
            "trace_capsules": {
                "trigger_state": trace_capsules.get("trigger_state"),
                "accepted_metadata_only": get_path(trace_capsules, ["summary", "accepted_metadata_only"], None),
                "quarantined": get_path(trace_capsules, ["summary", "quarantined"], None),
                "materialized_rows": get_path(trace_materialization, ["summary", "materialized_rows"], None),
                "raw_payload_rows": get_path(trace_materialization, ["summary", "raw_payload_rows"], None),
                "accepted_candidates_path": trace_capsules.get("accepted_candidates_path"),
                "materialized_rows_path": trace_materialization.get("rows_path"),
            },
            "adapter_bank_training_plan": {
                "trigger_state": legacy_adapter_plan.get("trigger_state"),
                "ready_for_zero_param_dry_run": legacy_adapter_plan.get("ready_for_zero_param_dry_run"),
                "ready_for_adapter_activation": legacy_adapter_plan.get("ready_for_adapter_activation"),
                "plan_rows": get_path(legacy_adapter_plan, ["summary", "plan_rows"], None),
                "selected_adapters": get_path(legacy_adapter_plan, ["summary", "selected_adapters"], []),
                "planned_adapter_lanes": get_path(legacy_adapter_plan, ["summary", "planned_adapter_lanes"], []),
                "zero_param_lanes": get_path(legacy_adapter_plan, ["summary", "zero_param_lanes"], []),
                "plan_path": legacy_adapter_plan.get("plan_path"),
            },
            "active_inference_pilot": {
                "trigger_state": active_inference_pilot.get("trigger_state"),
                "ready_for_world_model_training_signal": active_inference_pilot.get("ready_for_world_model_training_signal"),
                "mean_prediction_error": get_path(active_inference_pilot, ["summary", "mean_prediction_error"], None),
                "action_rankings": get_path(active_inference_pilot, ["summary", "action_rankings"], None),
                "accepted_belief_updates": get_path(active_inference_pilot, ["summary", "accepted_belief_updates"], None),
                "quarantined_belief_updates": get_path(active_inference_pilot, ["summary", "quarantined_belief_updates"], None),
                "replay_id": active_inference_pilot.get("replay_id"),
            },
        },
        "whitecell_threat_memory": {
            "trigger_state": whitecell.get("trigger_state"),
            "local_only": whitecell.get("local_only"),
            "active_block_and_escalate": whitecell_blockers,
        },
        "checkpoint_backup": {
            "policy_available": checkpoint_backup_policy.get("policy") == "project_theseus_checkpoint_backup_policy_v0",
            "last_status": checkpoint_backup.get("status"),
            "last_checkpoint_id": checkpoint_backup.get("checkpoint_id"),
            "github_enabled": get_path(checkpoint_backup_policy, ["providers", "github", "enabled"], None),
            "google_drive_enabled": get_path(checkpoint_backup_policy, ["providers", "google_drive", "enabled"], None),
        },
        "hive": {
            "policy_available": hive_policy.get("policy") == "project_theseus_hive_policy_v0",
            "node_name": hive_status.get("node_name"),
            "peer_count": hive_peers.get("peer_count"),
            "scheduler_nodes": get_path(hive_scheduler, ["summary", "nodes"], None),
            "best_training_node": get_path(hive_scheduler, ["summary", "best_training_node"], None),
            "best_inference_node": get_path(hive_scheduler, ["summary", "best_inference_node"], None),
        },
        "license": {
            "registration_complete": license_status.get("registration_complete"),
            "tier": get_path(license_status, ["entitlement", "tier"], None),
            "source": get_path(license_status, ["entitlement", "source"], None),
            "paid": get_path(license_status, ["entitlement", "paid"], None),
            "worker_chunks_allowed": license_worker_chunks.get("allowed"),
            "next_action": license_status.get("next_action"),
        },
        "checks": checks,
        "recommended_launch_command": launch_command(args.profile, macos=is_macos),
        "recommended_first_hours": [
            "Start with the requested allowed profile plus teacher proposal mode for architecture-wall diagnosis only.",
            "Keep legacy training, environment, and trace-fabric admission reports fresh before any long learning run.",
            "Let the daemon refresh arm lifecycle, resource, benchmark, RL, data, gate, checkpoint, and history reports each cycle.",
            "Generate the next mutated frontier before candidate promotion, then run the full candidate profile for matched gate evidence.",
        ],
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if (ready_for_autonomous_training or ready_for_local_macos_smoke_training) else 2


def check(gate: str, passed: bool, severity: str, evidence: str) -> dict[str, Any]:
    return {
        "gate": gate,
        "passed": bool(passed),
        "severity": severity,
        "evidence": evidence,
    }


def failed_candidate_gates(candidate: dict[str, Any]) -> list[str]:
    return [
        str(row.get("gate"))
        for row in candidate.get("checks", [])
        if isinstance(row, dict) and not row.get("passed")
    ]


def teacher_smoke_ok(report: dict[str, Any], calls: list[dict[str, Any]]) -> bool:
    return (
        report.get("status") == "completed"
        and returncode(report.get("returncode")) == 0
        and isinstance(report.get("response_json"), dict)
    ) or completed_teacher_calls(calls) > 0


def codex_cli_available(command: str) -> bool:
    if not command:
        return False
    try:
        result = subprocess.run(
            [command, "--version"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "codex" in (result.stdout + result.stderr).lower()


def long_run_guardrails_enabled(policy: dict[str, Any]) -> bool:
    safety_required = set((policy.get("safety") or {}).get("human_approval_required_for") or [])
    catalog_required = set(get_path(policy, ["online_source_catalog", "human_approval_required_for"], []))
    self_evolution = policy.get("self_evolution") or {}
    guarded_teacher_apply = (
        (policy.get("safety") or {}).get("autonomous_git_mutation")
        == "guarded_branch_and_gate_teacher_lane_only"
        and bool(self_evolution.get("enabled"))
        and bool(self_evolution.get("auto_apply_when_policy_allows"))
        and bool(self_evolution.get("requires_clean_worktree", True))
    )
    return (
        guarded_teacher_apply
        and "license_uncertain_source_import" in safety_required
        and "bulk_training_data_download" in safety_required
        and "commercial_game_or_rom_asset" in safety_required
        and "license_uncertain_source_import" in catalog_required
        and "bulk_training_data_download" in catalog_required
        and "commercial_game_or_rom_asset" in catalog_required
    )


def teacher_architecture_proposal_mode_available(
    teacher_policy: dict[str, Any], autonomy_policy: dict[str, Any]
) -> bool:
    allowed_reasons = set(teacher_policy.get("allowed_reasons") or [])
    critical_reasons = set(get_path(teacher_policy, ["budget", "critical_reasons_bypass_cooldown"], []))
    escalation = autonomy_policy.get("teacher_escalation") if isinstance(autonomy_policy.get("teacher_escalation"), dict) else {}
    trigger_wall_types = set(escalation.get("trigger_wall_types") or [])
    architecture_wall_trigger_available = bool(
        "architecture_wall" in trigger_wall_types
        or "architecture_training_wall" in trigger_wall_types
    )
    prompt_contract = (
        teacher_policy.get("teacher_prompt_contract")
        if isinstance(teacher_policy.get("teacher_prompt_contract"), dict)
        else {}
    )
    proposal_mode_blocks_training_rows = bool(
        prompt_contract.get("must_not_emit_training_rows")
        or prompt_contract.get("must_not_emit_training_rows_in_proposal_mode")
    )
    return (
        teacher_policy.get("default_mode") == "proposal"
        and teacher_policy.get("proposal_sandbox") == "read-only"
        and get_path(teacher_policy, ["budget", "distillation_training_enabled"], False) is True
        and get_path(teacher_policy, ["budget", "apply_mode_enabled"], True) is False
        and "architecture_wall" in allowed_reasons
        and "architecture_wall" in critical_reasons
        and architecture_wall_trigger_available
        and proposal_mode_blocks_training_rows
        and prompt_contract.get("must_keep_public_eval_payloads_heldout") is True
        and prompt_contract.get("must_route_training_distillation_to_governed_gate") is True
    )


def completed_teacher_calls(calls: list[dict[str, Any]]) -> int:
    return sum(
        1
        for call in calls
        if call.get("status") == "completed"
        and returncode(call.get("returncode")) == 0
        and isinstance(call.get("response_json"), dict)
    )


def active_whitecell_blockers(report: dict[str, Any], remediation: dict[str, Any] | None = None) -> list[str]:
    patterns = report.get("threat_patterns") if isinstance(report.get("threat_patterns"), list) else []
    resolved = resolved_whitecell_patterns(remediation or {})
    blockers = []
    for row in patterns:
        if not isinstance(row, dict):
            continue
        pattern_id = str(row.get("pattern_id") or "unknown_whitecell_pattern")
        if pattern_id in resolved:
            continue
        if row.get("active") and row.get("action") == "block_and_escalate":
            blockers.append(pattern_id)
    return sorted(set(blockers))


def resolved_whitecell_patterns(report: dict[str, Any]) -> set[str]:
    records = report.get("remediation_records") if isinstance(report.get("remediation_records"), list) else []
    resolved_statuses = {"remediated_decayed", "resolved", "inactive_memory_decayed"}
    return {
        str(row.get("pattern_id"))
        for row in records
        if isinstance(row, dict) and row.get("status") in resolved_statuses and row.get("safety_weakened") is False
    }


def returncode(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def public_payload_counters_zero(report: dict[str, Any]) -> bool:
    counters = get_path(report, ["summary", "public_payload_counters"], {})
    if not isinstance(counters, dict) or not counters:
        return False
    for value in counters.values():
        try:
            if int(value or 0) != 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def public_memory_prompt_calibration_clean(report: dict[str, Any]) -> bool:
    summary = dict_value(report.get("summary"))
    per_benchmark = dict_value(summary.get("per_benchmark"))
    public_boundary = dict_value(report.get("public_boundary"))
    longmemeval = dict_value(per_benchmark.get("longmemeval"))
    longmemeval_systems = dict_value(longmemeval.get("memory_systems"))
    longmemeval_best_single_non_vcm = max(
        [
            float(row.get("pass_rate") or 0.0)
            for name, row in longmemeval_systems.items()
            if name != "vcm_graph_evidence_selector" and isinstance(row, dict)
        ],
        default=0.0,
    )
    ladder_confirmation = (
        int(summary.get("scored_item_count") or 0) >= 1000
        and {"8k_to_32k", "32k_to_128k", "128k_plus"}.issubset(set(dict_value(summary.get("per_length_bucket"))))
    )
    longmemeval_repair_confirmation = (
        int(summary.get("scored_item_count") or 0) >= 600
        and int(longmemeval.get("items") or 0) >= 200
        and float(longmemeval.get("vcm_on_pass_rate") or 0.0) > 0.055
        and float(longmemeval.get("vcm_on_pass_rate") or 0.0) > float(longmemeval.get("vcm_off_pass_rate") or 0.0)
        and float(longmemeval.get("vcm_on_pass_rate") or 0.0) >= longmemeval_best_single_non_vcm
    )
    three_family_confirmation = (
        {"ruler", "babilong", "longmemeval"}.issubset(set(per_benchmark))
        and int(dict_value(per_benchmark.get("ruler")).get("items") or 0) > 0
        and int(dict_value(per_benchmark.get("babilong")).get("items") or 0) > 0
        and int(longmemeval.get("items") or 0) > 0
        and (ladder_confirmation or longmemeval_repair_confirmation)
    )
    largest_admitted_confirmation = (
        int(summary.get("scored_item_count") or 0) >= 2000
        and {"ruler", "babilong"}.issubset(set(per_benchmark))
        and int(dict_value(per_benchmark.get("ruler")).get("items") or 0) >= 1000
        and int(dict_value(per_benchmark.get("babilong")).get("items") or 0) >= 500
        and ladder_confirmation
        and len(dict_value(summary.get("per_length_bucket"))) >= 4
    )
    five_family_confirmation = (
        int(summary.get("scored_item_count") or 0) >= 2000
        and {
            "ruler",
            "babilong",
            "infinitebench",
            "needlebench_opencompass",
            "longbench_v2",
        }.issubset(set(per_benchmark))
        and all(
            int(dict_value(per_benchmark.get(benchmark)).get("items") or 0) > 0
            for benchmark in [
                "ruler",
                "babilong",
                "infinitebench",
                "needlebench_opencompass",
                "longbench_v2",
            ]
        )
        and len(dict_value(summary.get("per_length_bucket"))) >= 4
    )
    return (
        report.get("policy") == "project_theseus_vcm_public_memory_prompt_calibration_v1"
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and report.get("calibration_mode") == "prompt_level_public_memory_quarantined_slice"
        and int(summary.get("scored_item_count") or 0) > 0
        and (three_family_confirmation or largest_admitted_confirmation or five_family_confirmation)
        and all(int(value or 0) == 0 for value in dict_value(summary.get("forbidden_overlap_counts")).values())
        and public_boundary.get("public_payloads_quarantined") is True
        and bool(get_path(report, ["quarantine", "item_manifest"], ""))
        and bool(get_path(report, ["quarantine", "item_manifest_hash"], ""))
        and public_boundary.get("public_training_use_allowed") is False
        and int(public_boundary.get("public_rows_admitted_to_training") or 0) == 0
        and public_boundary.get("external_inference_allowed") is False
        and public_boundary.get("fallback_returns_allowed") is False
        and public_boundary.get("teacher_solving_allowed") is False
        and int(summary.get("external_inference_calls") or 0) == 0
        and int(summary.get("public_training_rows_written") or 0) == 0
        and int(summary.get("fallback_return_count") or 0) == 0
        and int(summary.get("teacher_solving_calls") or 0) == 0
    )


def public_memory_prompt_positive_transfer(report: dict[str, Any]) -> bool:
    summary = dict_value(report.get("summary"))
    scored = int(summary.get("scored_item_count") or 0)
    vcm_on = float(summary.get("vcm_on_pass_rate") or 0.0)
    vcm_off = float(summary.get("vcm_off_pass_rate") or 0.0)
    off_only = int(get_path(summary, ["win_counts", "vcm_off"], 0) or 0)
    delta = float(summary.get("vcm_over_flat_tail_delta") or (vcm_on - vcm_off))
    best_delta = float(summary.get("vcm_over_best_non_vcm_delta") or 0.0)
    if not public_memory_prompt_calibration_clean(report):
        return False
    if scored >= 100:
        return vcm_on > vcm_off and delta > 0.0 and best_delta > 0.0
    return vcm_on >= vcm_off and off_only == 0


def public_memory_readiness_clean(report: dict[str, Any]) -> bool:
    gates = [row for row in list_value(report.get("gates")) if isinstance(row, dict)]
    contamination = dict_value(report.get("contamination_counters"))
    private_summary = dict_value(report.get("private_analogue_summary"))
    coverage = dict_value(report.get("private_vcm_coverage"))
    return (
        report.get("policy") == "project_theseus_vcm_public_memory_readiness_audit_v1"
        and report.get("trigger_state") == "GREEN"
        and report.get("public_calibration_allowed") is True
        and gates
        and all(row.get("passed") is True for row in gates if row.get("severity") == "blocker")
        and all(int(value or 0) == 0 for value in contamination.values())
        and int(private_summary.get("off_only_wins") or 0) == 0
        and float(private_summary.get("vcm_on_pass_rate") or 0.0) >= 1.0
        and coverage.get("required_public_memory_residual_categories_present") is True
        and float(coverage.get("ablation_answer_lift") or 0.0) > 0.0
    )


def public_memory_private_repair_clean(report: dict[str, Any]) -> bool:
    has_repair_fixtures = (
        int(report.get("fixture_count") or 0) > 0
        and isinstance(report.get("residual_categories"), list)
        and bool(report.get("residual_categories"))
    )
    no_repair_needed = (
        report.get("repair_needed") is False
        and int(report.get("fixture_count") or 0) == 0
        and isinstance(report.get("residual_categories"), list)
        and not bool(report.get("residual_categories"))
    )
    return (
        report.get("policy") == "project_theseus_vcm_public_memory_private_residual_repair_v1"
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and report.get("private_only") is True
        and int(report.get("public_prompt_chars") or 0) == 0
        and int(report.get("public_answer_chars") or 0) == 0
        and int(report.get("public_training_rows_written") or 0) == 0
        and int(report.get("external_inference_calls") or 0) == 0
        and int(report.get("fallback_return_count") or 0) == 0
        and (has_repair_fixtures or no_repair_needed)
    )


def longmemeval_private_residual_clean(report: dict[str, Any]) -> bool:
    summary = dict_value(report.get("summary"))
    proposal = dict_value(report.get("future_public_calibration_proposal"))
    no_cheat = all(
        int(report.get(key) or 0) == 0
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
    return (
        report.get("policy") == "project_theseus_vcm_longmemeval_private_residual_curriculum_v1"
        and report.get("trigger_state") == "GREEN"
        and report.get("private_only") is True
        and int(summary.get("case_count") or 0) >= 150
        and float(summary.get("vcm_on_pass_rate") or 0.0) >= 0.85
        and float(summary.get("minimum_major_question_type_pass_rate") or 0.0) >= 0.75
        and float(summary.get("vcm_over_best_single_non_vcm_delta") or 0.0) >= 0.05
        and float(summary.get("vcm_on_evidence_recall") or 0.0) >= 0.85
        and len(list_value(report.get("hard_failures"))) == 0
        and proposal.get("proposal_state") == "READY_TO_PROPOSE_EXACT_ONCE_PUBLIC_CONFIRMATION"
        and proposal.get("run_public_automatically") is False
        and no_cheat
    )


def vcm_evidence_gauntlet_clean(report: dict[str, Any]) -> bool:
    summary = dict_value(report.get("summary"))
    abstention = dict_value(summary.get("abstention"))
    proposal = dict_value(report.get("public_confirmation_manifest_proposal"))
    no_cheat = all(
        int(report.get(key) or summary.get(key) or 0) == 0
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
    return (
        report.get("policy") == "project_theseus_vcm_evidence_gauntlet_v1"
        and report.get("trigger_state") == "GREEN"
        and report.get("private_only") is True
        and int(summary.get("case_count") or 0) >= 1000
        and float(summary.get("vcm_on_pass_rate") or 0.0) >= 0.90
        and float(summary.get("minimum_major_family_pass_rate") or 0.0) >= 0.80
        and float(summary.get("vcm_over_best_single_non_vcm_delta") or 0.0) >= 0.05
        and float(summary.get("vcm_on_evidence_recall") or 0.0) >= 0.90
        and float(abstention.get("precision") or 0.0) >= 0.95
        and float(abstention.get("recall") or 0.0) >= 0.95
        and len(list_value(report.get("hard_failures"))) == 0
        and proposal.get("run_public_automatically") is False
        and no_cheat
    )


def vcm_hard_memory_private_clean(report: dict[str, Any]) -> bool:
    summary = dict_value(report.get("summary"))
    abstention = dict_value(summary.get("abstention"))
    no_cheat = all(
        int(report.get(key) or summary.get(key) or 0) == 0
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
    return (
        report.get("policy") == "project_theseus_vcm_hard_memory_private_analogues_v1"
        and report.get("trigger_state") == "GREEN"
        and report.get("private_only") is True
        and int(summary.get("case_count") or 0) >= 1000
        and int(summary.get("family_count") or 0) >= 8
        and int(summary.get("length_bucket_count") or 0) >= 3
        and float(summary.get("vcm_on_pass_rate") or 0.0) >= 0.85
        and float(summary.get("minimum_family_pass_rate") or 0.0) >= 0.70
        and float(summary.get("vcm_over_best_single_non_vcm_delta") or 0.0) >= 0.03
        and float(summary.get("vcm_on_evidence_recall") or 0.0) >= 0.80
        and float(abstention.get("precision") or 0.0) >= 0.95
        and float(abstention.get("recall") or 0.0) >= 0.95
        and len(list_value(report.get("hard_failures"))) == 0
        and no_cheat
    )


def vcm_hard_memory_readiness_recorded(report: dict[str, Any]) -> bool:
    summary = dict_value(report.get("summary"))
    counters = dict_value(summary.get("public_payload_counters"))
    no_public_payloads = all(int(counters.get(key) or 0) == 0 for key in counters)
    return (
        report.get("policy") == "project_theseus_vcm_hard_memory_benchmark_readiness_v1"
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(summary.get("metadata_ready_count") or 0) >= 5
        and bool(summary.get("private_row_target_met")) is True
        and no_public_payloads
        and int(summary.get("fallback_return_count") or 0) == 0
        and int(summary.get("teacher_solving_calls") or 0) == 0
        and int(summary.get("external_inference_calls") or 0) == 0
    )


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def launch_command(profile: str, *, macos: bool = False) -> str:
    if macos:
        return (
            "python3 scripts/autonomy_cycle.py "
            f"--profile {profile} --execute --allow-teacher --out reports/autonomy_cycle_last.json"
        )
    return (
        "powershell -ExecutionPolicy Bypass -File scripts\\start_sparkstream.ps1 "
        f"-Profile {profile} -Execute -AllowTeacher -StartDaemon"
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
