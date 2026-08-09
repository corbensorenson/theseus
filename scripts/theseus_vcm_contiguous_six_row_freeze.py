#!/usr/bin/env python3
"""Freeze one contiguous, parent-only six-row VCM canary instrument.

The producer deliberately treats qualification artifacts as opaque hash-bound
prerequisites.  It never parses a target snapshot, target diff, evaluator, or
answer-bearing qualification report.  Hidden evidence is checked by the
role-separated audit owner.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_parent_only_materializer as parent_only  # noqa: E402

POLICY = "project_theseus_vcm_contiguous_six_row_freeze_v1"
CONFIG_POLICY = "project_theseus_vcm_contiguous_six_row_freeze_config_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_contiguous_six_row_freeze.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--store-out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report, store = freeze(path)
    store_path = p2a.resolve(args.store_out or str(cfg.get("store_out") or ""))
    report_path = p2a.resolve(args.out or str(cfg.get("report") or ""))
    p2a.write_json(store_path, store)
    report["store_artifact"] = identity(store_path)
    p2a.write_json(report_path, report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def freeze(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != CONFIG_POLICY:
        faults.append("config_policy_invalid")
    validate_binding(cfg, "owner", "owner_sha256", Path(__file__).resolve(), faults)
    validate_binding(
        cfg,
        "parent_only_owner",
        "parent_only_owner_sha256",
        Path(parent_only.__file__).resolve(),
        faults,
    )
    validate_binding(
        cfg,
        "test_owner",
        "test_owner_sha256",
        ROOT / "tests" / "test_theseus_vcm_contiguous_six_row_freeze.py",
        faults,
    )
    for binding in p2a.dicts(cfg.get("opaque_prerequisite_bindings")):
        source = p2a.resolve(str(binding.get("path") or ""))
        if not source.is_file() or p2a.sha256_file(source) != binding.get("sha256"):
            faults.append(f"opaque_prerequisite_binding_invalid:{binding.get('id')}")

    claim_binding = p2a.mapping(cfg.get("claim_instrument"))
    claim_path = p2a.resolve(str(claim_binding.get("path") or ""))
    if not claim_path.is_file() or p2a.sha256_file(claim_path) != claim_binding.get("sha256"):
        faults.append("claim_instrument_binding_invalid")
        claim = {}
    else:
        claim = p2a.read_json(claim_path)
    contract = freeze_contract(cfg, claim, faults)

    store_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for binding in p2a.dicts(cfg.get("rows")):
        built, row_faults = parent_only.build_row(binding, path, cfg)
        faults.extend(row_faults)
        request_id = str(built.get("request_id") or "")
        if not request_id or request_id in seen_ids:
            faults.append("request_id_invalid_or_duplicate")
        seen_ids.add(request_id)
        store_rows.append(p2a.mapping(built.get("store_row")))
        report_rows.append(p2a.mapping(built.get("report_row")))

    expected = int(cfg.get("expected_row_count") or 0)
    if expected != 6 or len(report_rows) != expected:
        faults.append("six_row_denominator_invalid")
    ready = not faults
    store = {
        "policy": parent_only.STORE_POLICY,
        "created_utc": p2a.now(),
        "campaign": "theseus_vcm_k2_05_contiguous_six_row_freeze_v1",
        "source_boundary": "exact_immutable_parent_archives_only",
        "content_storage": "archive_backed_no_duplicate_payload",
        "selector_policy": parent_only.SELECTOR_POLICY,
        "candidate_projection_only": True,
        "rows": store_rows,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "parent_target_or_evaluator_executions": 0,
    }
    report = {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if ready else "RED",
        "state": (
            "K2_05_CONTIGUOUS_SIX_ROW_INSTRUMENT_FROZEN"
            if ready
            else "K2_05_CONTIGUOUS_SIX_ROW_INSTRUMENT_FREEZE_FAILED"
        ),
        "faults": sorted(set(faults)),
        "config": identity(path),
        "row_count": len(report_rows),
        "regular_file_count": sum(int(row.get("regular_file_count") or 0) for row in report_rows),
        "text_page_count": sum(int(row.get("text_page_count") or 0) for row in report_rows),
        "candidate_visible_field_count": sum(
            len(p2a.mapping(row.get("candidate_surface"))) for row in report_rows
        ),
        "rows": report_rows,
        "frozen_contract": contract,
        "information_flow": {
            "producer_parsed_hidden_qualification_evidence": False,
            "producer_read_target_archive": False,
            "producer_read_target_diff": False,
            "producer_read_hidden_evaluator": False,
            "producer_read_answer_identifying_metadata": False,
            "packet_inputs": [
                "natural_language_request_utf8",
                "exact_parent_archive_regular_file_paths_and_bytes",
                "fixed_parent_only_selector_policy",
                "prospectively_frozen_arm_contract",
            ],
            "target_derived_effect_paths_present": False,
            "broad_parent_effect_root": "repository",
        },
        "panel_admitted": False,
        "candidate_or_control_calls": 0,
        "local_model_calls": 0,
        "external_reference_calls": 0,
        "teacher_calls": 0,
        "parent_target_or_evaluator_executions": 0,
        "network_calls": 0,
        "project_selected_quality_token_cap": None,
        "maximum_inference": cfg.get("maximum_inference"),
    }
    return report, store


def freeze_contract(
    cfg: dict[str, Any], claim: dict[str, Any], faults: list[str]
) -> dict[str, Any]:
    design = p2a.mapping(claim.get("prospective_design"))
    completion = p2a.mapping(claim.get("completion_policy"))
    reference = p2a.mapping(claim.get("openai_measurement_reference"))
    candidate_fields = p2a.strings(design.get("candidate_visible_fields"))
    expected_fields = [
        "natural_language_request",
        "callable_signature_when_present",
        "broad_parent_effect_root",
        "arm_specific_model_visible_context",
    ]
    if candidate_fields != expected_fields:
        faults.append("candidate_visible_field_contract_invalid")
    control_arms = p2a.dicts(design.get("control_qualification_arms"))
    if {str(row.get("id")) for row in control_arms} != {
        "local_no_added_context",
        "local_vcm_correct",
        "local_information_matched_plain_context",
        "local_maximal_ungoverned_context",
        "local_vcm_shuffled",
    }:
        faults.append("control_arm_contract_invalid")
    if (
        completion.get("project_selected_quality_token_cap") is not None
        or completion.get("normal_completion") != ["complete_artifact", "model_eos"]
        or completion.get("physical_context_boundary_hit_invalidates_observation") is not True
    ):
        faults.append("completion_contract_invalid")
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
        faults.append("subscription_only_reference_contract_invalid")
    authority = p2a.mapping(cfg.get("authority"))
    if authority != {
        "local_model_calls_authorized": 0,
        "external_reference_calls_authorized": 0,
        "teacher_calls_authorized": False,
        "training_rows_authorized": False,
        "serving_authorized": False,
        "D1_authorized": False,
        "D2_authorized": False,
        "book_support_promotion_authorized": False,
        "user_or_operator_gate": False,
    }:
        faults.append("authority_contract_invalid")
    stop_rules = p2a.strings(cfg.get("k3_stop_rules"))
    invalidations = p2a.dicts(cfg.get("invalidation_classes"))
    if len(stop_rules) < 8 or len(invalidations) < 8:
        faults.append("k3_stop_or_invalidation_contract_incomplete")
    cost = p2a.mapping(cfg.get("cost_custody"))
    if (
        cost.get("openai_api_spend_authorized_usd") != 0
        or cost.get("reference_cost_denominator") != "separate_never_mixed_with_local"
        or cost.get("record_actual_tokens_time_verifier_work_and_total_system_cost") is not True
    ):
        faults.append("cost_custody_contract_invalid")
    return {
        "claim_id": claim.get("active_claim_id"),
        "instrument_role": "unpowered_six_row_real_work_canary_not_claim_denominator",
        "candidate_visible_fields": candidate_fields,
        "hidden_fields": design.get("hidden_fields"),
        "broad_parent_effect_root": p2a.mapping(design.get("effect_boundary")).get(
            "broad_parent_effect_root"
        ),
        "control_arms": control_arms,
        "pre_inference_interventions": design.get("pre_inference_interventions"),
        "primary_canary_outcome": cfg.get("primary_canary_outcome"),
        "secondary_outcomes": design.get("secondary_estimands"),
        "completion_policy": completion,
        "context_resource_policy": claim.get("context_resource_policy"),
        "openai_measurement_reference": reference,
        "cost_custody": cost,
        "invalidation_classes": invalidations,
        "k3_stop_rules": stop_rules,
        "authority": authority,
        "maximum_inference": cfg.get("contract_maximum_inference"),
    }


def validate_binding(
    cfg: dict[str, Any], path_key: str, hash_key: str, expected: Path, faults: list[str]
) -> None:
    actual = p2a.resolve(str(cfg.get(path_key) or ""))
    if actual != expected or not actual.is_file() or p2a.sha256_file(actual) != cfg.get(hash_key):
        faults.append(f"{path_key}_binding_invalid")


def identity(path: Path) -> dict[str, Any]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path), "bytes": path.stat().st_size}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "trigger_state",
            "state",
            "faults",
            "row_count",
            "regular_file_count",
            "text_page_count",
            "candidate_visible_field_count",
            "local_model_calls",
            "external_reference_calls",
        )
    }


if __name__ == "__main__":
    raise SystemExit(main())
