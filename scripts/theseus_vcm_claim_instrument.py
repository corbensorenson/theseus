#!/usr/bin/env python3
"""Call-free audit for the active Virtual Context ABI claim instrument."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_assistant_route_integrity_v2 as route
import vcm_consumer_abi as consumer_abi
import vcm_consumer_integration_gate as consumer_gate
import vcm_context_governor_gate as governor_gate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_claim_instrument.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_claim_instrument_audit.json"
POLICY = "project_theseus_vcm_claim_instrument_audit_v1"
CONFIG_POLICY = "project_theseus_vcm_claim_instrument_v1"
MODEL_CONTEXT_TOKENS = 262_144


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = audit(p2a.resolve(args.config))
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if (
        config.get("policy") != CONFIG_POLICY
        or config.get("state") != "PROSPECTIVE_VCM_INSTRUMENT_BINDING_ZERO_CALLS"
        or config.get("active_claim_id") != "virtual-context-abi.core"
    ):
        faults.append("config_identity_invalid")
    for row in p2a.dicts(config.get("source_bindings")):
        path = p2a.resolve(str(row.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != row.get("sha256"):
            faults.append(f"source_binding_invalid:{row.get('id') or p2a.rel(path)}")
    matrix = p2a.read_json(ROOT / "configs" / "roadmap_implementation_matrix.json")
    active = p2a.mapping(p2a.mapping(matrix.get("research_program_recenter")).get("active_claim"))
    if (
        active.get("claim_id") != "virtual-context-abi.core"
        or active.get("fresh_claim_pool_authorized") is not False
        or active.get("D1_authorized") is not False
    ):
        faults.append("active_claim_binding_invalid")
    terminal = p2a.read_json(ROOT / "reports" / "theseus_semantic_ir_production_adequacy_v6_terminal_disposition.json")
    if (
        terminal.get("trigger_state") != "GREEN"
        or terminal.get("scientific_status") != "INCONCLUSIVE_EXPERIMENT"
        or p2a.mapping(terminal.get("portfolio_transition")).get("next_claim_id")
        != "virtual-context-abi.core"
        or p2a.mapping(terminal.get("portfolio_transition")).get("next_stage_model_calls_authorized") != 0
    ):
        faults.append("predecessor_residual_binding_invalid")
    governor = governor_gate.build_report(
        governor_gate.DEFAULT_CONFIG,
        governor_gate.read_json(governor_gate.DEFAULT_CONFIG),
        time.perf_counter(),
    )
    consumer = consumer_gate.build_report()
    if (
        governor.get("trigger_state") != "GREEN"
        or p2a.mapping(governor.get("summary")).get("context_abi_fixture_status") != "ready"
        or p2a.mapping(governor.get("summary")).get("context_resolver_status") != "ready"
    ):
        faults.append("vcm_governor_mechanics_invalid")
    if (
        consumer.get("trigger_state") != "GREEN"
        or int(p2a.mapping(consumer.get("summary")).get("expected_invalid_rejected_count") or 0) != 5
        or int(p2a.mapping(consumer.get("summary")).get("ready_consumer_count") or 0) != 45
    ):
        faults.append("vcm_consumer_mechanics_invalid")
    packet_controls = packet_control_audit()
    if any(row.get("passed") is not True for row in packet_controls):
        faults.append("vcm_packet_control_invalid")
    route_controls = route_control_audit(config)
    if any(row.get("passed") is not True for row in route_controls):
        faults.append("vcm_route_control_invalid")
    design = p2a.mapping(config.get("prospective_design"))
    materializer_audit = p2a.read_json(
        ROOT / "reports" / "theseus_vcm_parent_only_materializer_audit.json"
    )
    if (
        materializer_audit.get("trigger_state") != "GREEN"
        or materializer_audit.get("state")
        != "K2_04_PARENT_ONLY_MATERIALIZER_ROLE_SEPARATELY_REDERIVED"
        or materializer_audit.get("candidate_or_control_calls") != 0
        or materializer_audit.get("external_reference_calls") != 0
        or not all(
            p2a.mapping(materializer_audit.get("conclusions")).get(key) is True
            for key in (
                "exact_parent_archives_only",
                "complete_parent_text_frontier_no_convenience_cap",
                "selector_inputs_request_and_parent_only",
                "candidate_visible_bytes_individually_rederived",
                "single_broad_parent_effect_root",
                "target_derived_effect_paths_absent",
                "matched_non_vcm_context_information_identity",
                "production_vcm_consumer_abi_ready",
            )
        )
    ):
        faults.append("parent_only_materializer_audit_invalid")
    if set(p2a.strings(design.get("candidate_visible_fields"))) != {
        "natural_language_request",
        "callable_signature_when_present",
        "broad_parent_effect_root",
        "arm_specific_model_visible_context",
    }:
        faults.append("candidate_visible_field_contract_invalid")
    effect_boundary = p2a.mapping(design.get("effect_boundary"))
    if (
        effect_boundary.get("broad_parent_effect_root") != "repository"
        or effect_boundary.get("same_root_for_every_arm") is not True
        or effect_boundary.get("target_derived_effect_paths_forbidden") is not True
        or effect_boundary.get("candidate_patch_scope_recomputed_independently") is not True
    ):
        faults.append("broad_parent_effect_boundary_invalid")
    control_arms = p2a.dicts(design.get("control_qualification_arms"))
    claim_arms = p2a.dicts(design.get("claim_arms"))
    required_control_arms = {
        "local_no_added_context", "local_vcm_correct",
        "local_information_matched_plain_context", "local_maximal_ungoverned_context",
        "local_vcm_shuffled",
    }
    required_claim_arms = {
        "local_vcm_correct", "local_frozen_strongest_control",
        "luna_vcm_correct", "luna_frozen_same_control",
    }
    interventions = set(p2a.strings(design.get("pre_inference_interventions")))
    power = p2a.mapping(design.get("power_design"))
    minimum_power = worst_case_exact_mcnemar_power(
        int(design.get("task_count") or 0),
        float(power.get("minimum_useful_absolute_effect") or 0.0),
        float(power.get("one_sided_alpha") or 0.0),
    )
    predecessor_power = worst_case_exact_mcnemar_power(
        int(design.get("task_count") or 0) - 1,
        float(power.get("minimum_useful_absolute_effect") or 0.0),
        float(power.get("one_sided_alpha") or 0.0),
    )
    if (
        {str(row.get("id") or "") for row in control_arms} != required_control_arms
        or {str(row.get("id") or "") for row in claim_arms} != required_claim_arms
        or interventions != {"omitted", "stale", "shuffled", "wrong_scope", "tainted", "revoked"}
        or int(design.get("control_qualification_task_count") or 0) != 9
        or int(design.get("task_count") or 0) != 53
        or design.get("task_count_binding_rule") != "derive_from_predeclared_useful_effect_power_analysis_before_source_acquisition"
        or float(power.get("minimum_useful_absolute_effect") or 0.0) != 0.35
        or float(power.get("one_sided_alpha") or 0.0) != 0.05
        or float(power.get("minimum_power") or 0.0) != 0.80
        or int(power.get("minimum_task_count_satisfying_worst_case_power") or 0) != 53
        or abs(float(power.get("worst_case_power_at_53") or 0.0) - minimum_power) > 1e-12
        or power.get("task_count_52_fails_minimum_power") is not True
        or minimum_power < 0.80
        or predecessor_power >= 0.80
        or design.get("hidden_evaluation_after_all_candidate_seals") is not True
        or design.get("source_disjoint_from_all_prior_claim_and_adequacy_denominators") is not True
        or design.get("control_and_claim_panels_source_disjoint") is not True
    ):
        faults.append("prospective_design_invalid")
    completion = p2a.mapping(config.get("completion_policy"))
    if (
        completion.get("project_selected_quality_token_cap") is not None
        or completion.get("normal_completion") != ["complete_artifact", "model_eos"]
        or completion.get("physical_context_boundary_hit_invalidates_observation") is not True
        or completion.get("host_safety_activation_is_not_capability_failure") is not True
        or completion.get("context_materialization_budget_is_causal_resource") is not True
    ):
        faults.append("completion_policy_invalid")
    reference = p2a.mapping(config.get("openai_measurement_reference"))
    if (
        reference.get("provider") != "OpenAI"
        or reference.get("model") != "gpt-5.6-luna"
        or reference.get("reasoning_effort") != "xhigh"
        or reference.get("transport") != "demonstrably_codex_subscription_backed_only"
        or reference.get("billable_api_spend_authorized") is not False
        or reference.get("api_credentials_or_api_billing_route_allowed") is not False
        or reference.get("subscription_provenance_receipt_required_before_authorization") is not True
        or reference.get("omit_if_subscription_provenance_cannot_be_proved") is not True
    ):
        faults.append("openai_reference_subscription_boundary_invalid")
    authority = p2a.mapping(config.get("authority"))
    if (
        authority.get("local_model_calls_authorized") != 0
        or authority.get("external_reference_calls_authorized") != 0
        or authority.get("teacher_calls_authorized") is not False
        or authority.get("training_rows_authorized") is not False
        or authority.get("D1_authorized") is not False
        or authority.get("D2_authorized") is not False
        or authority.get("book_support_promotion_authorized") is not False
        or authority.get("user_or_operator_gate") is not False
    ):
        faults.append("cross_stage_authority_present")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "active_claim_id": "virtual-context-abi.core",
        "config": artifact(config_path),
        "predecessor_terminal_disposition": artifact(ROOT / "reports" / "theseus_semantic_ir_production_adequacy_v6_terminal_disposition.json"),
        "governor_replay": {
            "trigger_state": governor.get("trigger_state"),
            "summary": governor.get("summary"),
        },
        "consumer_replay": {
            "trigger_state": consumer.get("trigger_state"),
            "summary": consumer.get("summary"),
        },
        "parent_only_materializer_audit": {
            "trigger_state": materializer_audit.get("trigger_state"),
            "state": materializer_audit.get("state"),
            "conclusions": materializer_audit.get("conclusions"),
        },
        "packet_controls": packet_controls,
        "route_controls": route_controls,
        "prospective_design_ready": not faults,
        "task_source_acquisition_opened": False,
        "candidate_generation_opened": False,
        "hidden_evaluation_opened": False,
        "counters": zero_counters(),
        "faults": sorted(set(faults)),
        "maximum_inference": "A GREEN audit proves only that the existing VCM packet, consumer, invalid-control, and model-prompt route mechanics support prospective experimental binding. It authorizes zero model or reference calls and establishes no VCM usefulness, safety advantage, task competence, D1 result, training value, serving value, or ASI Stack support.",
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def packet_control_audit() -> list[dict[str, Any]]:
    index_path = ROOT / "reports" / "virtual_context_memory_index.json"
    index = p2a.read_json(index_path)
    pages = [
        row for row in p2a.dicts(index.get("pages"))
        if row.get("status") == "active"
        and row.get("model_visible") is True
        and not p2a.strings(row.get("taints"))
        and p2a.resolve(str(row.get("source_path") or "")).is_file()
    ]
    if not pages:
        return [{"control": "known_positive_source", "passed": False, "faults": ["no_active_untainted_source"]}]
    page = pages[0]
    source = p2a.resolve(str(page.get("source_path") or ""))
    now = datetime(2026, 8, 3, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-claim-audit-") as directory:
        output = Path(directory) / "candidate.json"

        def packet(**overrides: Any) -> dict[str, Any]:
            args: dict[str, Any] = {
                "consumer_id": "theseus.vcm.claim.candidate",
                "purpose": "claim_candidate_context",
                "read_set": [str(source)],
                "write_set": [str(output)],
                "authority_ceiling": ["local_read"],
                "permitted_uses": ["candidate_context_read"],
                "governor_path": consumer_abi.DEFAULT_GOVERNOR,
                "semantic_index_path": index_path,
                "context_refs": [{"kind": "semantic_address", "ref": page.get("address"), "required": True}],
                "now_utc": now,
            }
            args.update(overrides)
            return consumer_abi.build_consumer_packet(**args)

        positive = packet()
        controls: list[tuple[str, dict[str, Any], bool, str]] = [
            ("correct", positive, True, ""),
            ("omitted", packet(read_set=[str(Path(directory) / "missing.json")], context_refs=[]), False, "CONTEXT_REQUIRED_MISSING"),
            ("stale", packet(context_refs=[{"ref": str(source), "exists": True, "required": True, "created_utc": "2026-08-01T00:00:00Z", "max_age_seconds": 60}]), False, "CONTEXT_REQUIRED_STALE"),
            ("tainted", packet(taint_labels=["public_benchmark_payload"]), False, "CONTEXT_TAINT_DENIED"),
            ("revoked", packet(context_refs=[{"ref": str(source), "exists": True, "required": True, "revoked": True}]), False, "CONTEXT_REVOKED_OR_DELETED"),
            ("wrong_scope", packet(context_refs=[{"kind": "semantic_address", "ref": "vcm://theseus/wrong/scope@v1", "required": True}]), False, "CONTEXT_SEMANTIC_REF_UNRESOLVED"),
            ("over_compressed", packet(compression_loss=0.6, max_compression_loss=0.35), False, "CONTEXT_OVER_COMPRESSED"),
        ]
        rows = []
        for name, value, ready, expected in controls:
            faults = p2a.strings(value.get("typed_faults"))
            passed = value.get("ready") is ready and (not expected or expected in faults)
            rows.append({"control": name, "passed": passed, "ready": value.get("ready"), "expected_fault": expected, "faults": faults})
        return rows


def route_control_audit(config: dict[str, Any]) -> list[dict[str, Any]]:
    identity = p2a.mapping(config.get("frozen_local_model"))
    context_a = "The task requires the repository's frozen policy value ALPHA and its exact source-bound version."
    context_b = "The dependent caller must preserve the BETA invariant when applying that policy."
    address_a = "vcm://theseus/claim-fixture/policy@v1"
    address_b = "vcm://theseus/claim-fixture/caller@v1"
    certificate = "cert:vcm-claim-fixture"
    selected = {"ready": True, "selected_pages": [{"address": address_a}, {"address": address_b}]}
    compiled = {"model_visible_pages": [
        {"address": address_a, "certificate_id": certificate, "execution_class": "authorized_task_state", "taints": [], "materialized_text": context_a},
        {"address": address_b, "certificate_id": certificate, "execution_class": "authorized_task_state", "taints": [], "materialized_text": context_b},
    ]}
    common = {
        "execution_mode": route.INTEGRATED_MODE,
        "prompt": "Apply the repository policy while preserving its dependent invariant.",
        "model_identity": identity,
        "reflexive_dispatch": dispatch_fixture(),
        "reflexive_verification": {"state": "VERIFIED"},
        "structured_execution": {"active": False, "terminal_outcome": "not_required"},
    }
    correct = route.build_generation_request(selected_context=selected, compiled_context=compiled, **common)
    omitted = route.build_generation_request(selected_context=selected, compiled_context={"model_visible_pages": []}, **common)
    tainted_compiled = copy.deepcopy(compiled)
    tainted_compiled["model_visible_pages"][0]["taints"] = ["public_benchmark_payload"]
    tainted = route.build_generation_request(selected_context=selected, compiled_context=tainted_compiled, **common)
    reversed_selected = {"ready": True, "selected_pages": list(reversed(selected["selected_pages"]))}
    shuffled = route.build_generation_request(selected_context=reversed_selected, compiled_context=compiled, **common)
    plain_text = context_a + "\n" + context_b
    plain = route.build_generation_request(
        execution_mode=route.DIRECT_MODE,
        prompt=common["prompt"] + "\n\n[plain_context]\n" + plain_text,
        model_identity=identity,
    )
    no_context = route.build_generation_request(
        execution_mode=route.DIRECT_MODE,
        prompt=common["prompt"],
        model_identity=identity,
    )
    maximal = route.build_generation_request(
        execution_mode=route.DIRECT_MODE,
        prompt=common["prompt"] + "\n\n[maximal_ungoverned_context]\n" + plain_text + "\n" + plain_text,
        model_identity=identity,
    )
    return [
        {"control": "correct_vcm", "passed": correct.get("ready") is True and len(p2a.dicts(p2a.mapping(correct.get("binding")).get("context_pages"))) == 2},
        {"control": "omitted_materialization", "passed": omitted.get("ready") is False and "selected_vcm_content_not_consumable" in p2a.strings(omitted.get("faults"))},
        {"control": "tainted_materialization", "passed": tainted.get("ready") is False and any(str(fault).startswith("selected_vcm_page_taint_denied") for fault in p2a.strings(tainted.get("faults")))},
        {"control": "shuffled_packet_changes_binding", "passed": shuffled.get("ready") is True and p2a.mapping(shuffled.get("binding")).get("model_prompt_sha256") != p2a.mapping(correct.get("binding")).get("model_prompt_sha256")},
        {"control": "information_matched_plain_context", "passed": plain.get("ready") is True and plain_text in str(plain.get("model_prompt") or "") and not p2a.dicts(p2a.mapping(plain.get("binding")).get("context_pages"))},
        {"control": "no_added_context", "passed": no_context.get("ready") is True and not p2a.dicts(p2a.mapping(no_context.get("binding")).get("context_pages"))},
        {"control": "maximal_ungoverned_context", "passed": maximal.get("ready") is True and len(str(maximal.get("model_prompt") or "")) > len(str(plain.get("model_prompt") or ""))},
        {"control": "raw_context_excluded_from_receipt", "passed": plain_text not in json.dumps(correct.get("binding"), sort_keys=True) and p2a.mapping(correct.get("binding")).get("raw_context_text_stored") is False},
    ]


def dispatch_fixture() -> dict[str, Any]:
    return {
        "trace_id": "trace:vcm-claim-audit",
        "decision_digest": "d" * 64,
        "selection": {"selected_proposal_ids": ["proposal:vcm"], "terminal_outcome": "prepared"},
        "proposals": [{"proposal_id": "proposal:vcm", "capability_id": "assistant.chat_checkpoint", "capability_ids": ["assistant.chat_checkpoint"]}],
        "plan_nodes": [{"node_id": "node:vcm", "capability_id": "assistant.chat_checkpoint", "dependencies": [], "deadline_ms": 30_000}],
        "effort": {"realized_limits": {"deadline_ms": 30_000, "maximum_parallelism": 1}},
    }


def worst_case_exact_mcnemar_power(task_count: int, effect: float, alpha: float) -> float:
    if task_count <= 0 or effect <= 0.0 or alpha <= 0.0:
        return 0.0
    lower = effect
    upper = 1.0
    intervals = 256
    step = (upper - lower) / intervals
    points = [lower + index * step for index in range(intervals + 1)]
    values = [
        exact_mcnemar_power(task_count, discordance, effect, alpha)
        for discordance in points
    ]
    candidates = [values[0], values[-1]]
    for index in range(1, intervals):
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]:
            candidates.append(
                bounded_minimum(
                    lambda discordance: exact_mcnemar_power(
                        task_count, discordance, effect, alpha
                    ),
                    points[index - 1],
                    points[index + 1],
                )
            )
    return min(candidates)


def bounded_minimum(function: Any, lower: float, upper: float) -> float:
    """Deterministically minimize a smooth scalar function on a sealed bracket."""
    golden = (5.0**0.5 - 1.0) / 2.0
    left = upper - golden * (upper - lower)
    right = lower + golden * (upper - lower)
    left_value = function(left)
    right_value = function(right)
    while upper - lower > 1e-12:
        if left_value <= right_value:
            upper = right
            right = left
            right_value = left_value
            left = upper - golden * (upper - lower)
            left_value = function(left)
        else:
            lower = left
            left = right
            left_value = right_value
            right = lower + golden * (upper - lower)
            right_value = function(right)
    return min(left_value, right_value)


def exact_mcnemar_power(task_count: int, discordance: float, effect: float, alpha: float) -> float:
    from math import comb

    treatment_win_rate = (discordance + effect) / (2.0 * discordance)

    def tail(total: int, threshold: int, probability: float) -> float:
        return sum(
            comb(total, wins) * probability**wins * (1.0 - probability) ** (total - wins)
            for wins in range(threshold, total + 1)
        )

    def critical(total: int) -> int:
        for threshold in range(total + 1):
            if tail(total, threshold, 0.5) <= alpha:
                return threshold
        return total + 1

    result = 0.0
    for discordant_pairs in range(task_count + 1):
        probability_of_count = (
            comb(task_count, discordant_pairs)
            * discordance**discordant_pairs
            * (1.0 - discordance) ** (task_count - discordant_pairs)
        )
        threshold = critical(discordant_pairs)
        if threshold <= discordant_pairs:
            result += probability_of_count * tail(
                discordant_pairs, threshold, treatment_win_rate
            )
    return result


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


def zero_counters() -> dict[str, int]:
    return {
        "candidate_or_control_calls": 0, "local_model_calls": 0,
        "external_inference_calls": 0, "hidden_evaluator_executions": 0,
        "teacher_calls": 0, "training_rows_written": 0,
        "D1_cases_consumed": 0, "D2_cases_consumed": 0,
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "active_claim_id", "prospective_design_ready",
        "task_source_acquisition_opened", "candidate_generation_opened",
        "hidden_evaluation_opened", "faults", "counters",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
