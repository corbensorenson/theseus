#!/usr/bin/env python3
"""Freeze a fresh D1 source cohort from metadata only after a P4-v2r2-r2 survivor.

Archive contents and every parent, target, oracle, evaluator, candidate, and
control outcome are deliberately outside this selection boundary.  A selected
task that later fails materialization is retained and makes the qualification
inconclusive; it is never silently replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_d1_fresh_qualification_instrument as d1  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "theseus_d1_source_selection.json"
POLICY = "project_theseus_d1_metadata_only_source_selection_v1"
LEDGER_POLICY = "project_theseus_d1_online_metadata_frame_v1"
REGISTRY_POLICY = "project_theseus_d1_online_source_registry_v1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SOURCE_SUFFIXES = (".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".rs")
TEST_MARKERS = ("test", "tests", "spec", "specs")
FORBIDDEN_OUTCOME_KEYS = {
    "answer",
    "category",
    "candidate_output",
    "candidate_result",
    "control_output",
    "control_result",
    "evaluation",
    "evaluator_outcome",
    "expected",
    "hidden_tests",
    "oracle",
    "parent_passed",
    "required_constructs",
    "return_shape",
    "solution",
    "solution_body",
    "solution_expr",
    "source_task_id",
    "tests",
    "target_passed",
    "tests_passed",
    "type_family",
    "useful",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=d1.relative(DEFAULT_CONFIG))
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    config_path = d1.resolve(args.config)
    config = d1.read_json(config_path)
    report = build_report(config_path)
    report_path = d1.resolve(args.out or str(config["report"]))
    if args.freeze and report.get("registry_ready") is True:
        registry_path = d1.resolve(str(config["frozen_source_registry"]))
        write_json_exclusive(registry_path, dict(report["registry_candidate"]))
        report["registry_written"] = d1.artifact(registry_path)
    else:
        report["registry_written"] = {}
    d1.write_json(report_path, report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def build_report(
    config_path: Path = DEFAULT_CONFIG,
    *,
    disposition_override: dict[str, Any] | None = None,
    ledger_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = d1.read_json(config_path)
    faults = validate_config(config)
    instrument_path = d1.resolve(str(config.get("instrument") or ""))
    instrument_report = d1.build_report(
        instrument_path, disposition_override=disposition_override
    )
    activation_ready = instrument_report.get("source_acquisition_authorized") is True
    base = {
        "policy": "project_theseus_d1_metadata_only_source_selection_report_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "config": d1.artifact(config_path),
        "instrument_audit": instrument_report,
        "activation_state": instrument_report.get("activation_state"),
        "source_metadata_ledger_ingested": False,
        "source_acquisition_authorized": activation_ready and not faults,
        "registry_ready": False,
        "registry_candidate": {},
        "candidate_or_control_calls_authorized": False,
        "archive_fetch_authorized": False,
        "maximum_inference": str(config.get("maximum_inference") or ""),
    }
    if faults:
        return base
    if not activation_ready:
        base["trigger_state"] = "PAUSED"
        return base

    ledger_path = d1.resolve(str(config.get("candidate_metadata_ledger") or ""))
    if ledger_override is None and not ledger_path.is_file():
        base["trigger_state"] = "PAUSED"
        base["activation_state"] = "WAITING_FOR_COMPLETE_ONLINE_METADATA_FRAME"
        return base
    ledger = ledger_override if ledger_override is not None else d1.read_json(ledger_path)
    disposition_path = d1.resolve(
        str(d1.mapping(d1.read_json(instrument_path).get("activation")).get(
            "p4_terminal_disposition"
        ) or "")
    )
    disposition = (
        disposition_override
        if disposition_override is not None
        else d1.read_json(disposition_path)
    )
    disposition_sha256 = (
        d1.stable_hash(disposition)
        if disposition_override is not None
        else hashlib.sha256(disposition_path.read_bytes()).hexdigest()
    )
    ledger_faults = audit_ledger(
        ledger,
        config,
        disposition=disposition,
        disposition_sha256=disposition_sha256,
    )
    base["source_metadata_ledger_ingested"] = True
    base["metadata_frame"] = metadata_frame_identity(ledger, ledger_path, ledger_override)
    prior_repositories, prior_faults = d1.prior_repository_inventory(
        d1.read_json(instrument_path)
    )
    ledger_faults.extend(prior_faults)
    temporal_guard, temporal_faults = audit_temporal_guard(config)
    ledger_faults.extend(temporal_faults)
    base["temporal_contamination_guard"] = temporal_guard
    benchmark_repositories = {
        str(value)
        for value in temporal_guard.get("public_benchmark_repositories", [])
        if isinstance(value, str)
    }
    cohort_size = int(
        d1.mapping(d1.read_json(instrument_path).get("power_design")).get(
            "design_derived_cohort_size"
        )
        or 0
    )
    selected, exclusions = select_rows(
        dictionaries(ledger.get("rows")),
        config=config,
        campaign_id=str(d1.read_json(instrument_path).get("campaign_id") or ""),
        prior_repositories=set(prior_repositories),
        cohort_size=cohort_size,
        model_snapshot_observed_utc=str(
            temporal_guard.get("model_snapshot_observed_utc") or ""
        ),
        public_benchmark_repositories=benchmark_repositories,
    )
    if len(selected) != cohort_size:
        ledger_faults.append(
            f"eligible_distinct_repository_count_below_design:{len(selected)}/{cohort_size}"
        )
    base["faults"] = sorted(set(ledger_faults))
    base["trigger_state"] = "GREEN" if not ledger_faults else "PAUSED"
    base["selection"] = {
        "design_derived_cohort_size": cohort_size,
        "selected_count": len(selected),
        "excluded_count": len(exclusions),
        "prior_repository_count": len(prior_repositories),
        "prior_repositories_sha256": d1.stable_hash(prior_repositories),
        "selection_order": d1.mapping(config.get("selection")).get("order"),
        "selected_identity_sha256": d1.stable_hash(
            [source_identity(row) for row in selected]
        ),
        "exclusion_reason_counts": reason_counts(exclusions),
    }
    if ledger_faults:
        return base
    registry = {
        "policy": REGISTRY_POLICY,
        "state": "FIXED_BEFORE_ARCHIVE_FETCH_PARENT_TARGET_ORACLE_EVALUATOR_OR_CANDIDATE_EXECUTION",
        "campaign_id": d1.read_json(instrument_path).get("campaign_id"),
        "claim_id": d1.read_json(instrument_path).get("claim_id"),
        "instrument": d1.artifact(instrument_path),
        "P4V2R2R2_terminal_disposition": instrument_report.get("p4_terminal_disposition"),
        "metadata_frame": base["metadata_frame"],
        "selection_config": d1.artifact(config_path),
        "task_count": len(selected),
        "distinct_repository_count": len({row["repository"].lower() for row in selected}),
        "tasks": [
            {"campaign_index": index, **row}
            for index, row in enumerate(selected, 1)
        ],
        "boundaries": {
            "archive_fetches": 0,
            "parent_target_oracle_or_evaluator_executions": 0,
            "candidate_or_control_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "user_task_label_or_approval_dependency": False,
        },
        "replacement_after_membership_freeze": False,
        "materialization_failure_disposition": (
            "INCONCLUSIVE_EXPERIMENT_NOT_REPLACED"
        ),
        "source_disjoint_from": {
            "prior_P2_P3_P4_P4R_P4S_P4V2R2_P4V2R2R2_repositories": d1.stable_hash(
                prior_repositories
            ),
            "public_benchmark_catalog_repositories": d1.stable_hash(
                sorted(benchmark_repositories)
            ),
            "training": "all_selected_D1_tasks_permanently_excluded",
            "D2": "independent_neural_surface",
        },
        "maximum_inference": str(config.get("maximum_inference") or ""),
    }
    base["registry_ready"] = True
    base["registry_candidate"] = registry
    base["archive_fetch_authorized"] = False
    return base


def validate_config(config: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("selection_policy_invalid")
    if config.get("state") != "BOUND_BEFORE_P4_SURVIVOR_AND_D1_METADATA_ACQUISITION":
        faults.append("selection_state_invalid")
    discovery = d1.mapping(config.get("discovery_frame"))
    for key in (
        "external_inference_calls",
        "teacher_calls",
        "candidate_or_control_calls",
        "parent_target_oracle_or_evaluator_executions",
        "archive_fetches",
    ):
        if int(discovery.get(key) or 0) != 0:
            faults.append(f"preselection_boundary_nonzero:{key}")
    if discovery.get("require_every_partition_complete") is not True:
        faults.append("complete_query_partitions_not_required")
    selection = d1.mapping(config.get("selection"))
    if selection.get("membership_fixed_before_archive_fetch") is not True:
        faults.append("membership_not_fixed_before_archive_fetch")
    if selection.get("replacement_after_membership_freeze") is not False:
        faults.append("postfreeze_replacement_allowed")
    if selection.get("user_or_operator_approval_required") is not False:
        faults.append("user_or_operator_gate_present")
    authority = d1.mapping(config.get("authority"))
    if authority.get(
        "source_metadata_acquisition_opens_only_after_green_P4V2R2R2_survivor"
    ) is not True:
        faults.append("P4V2R2R2_survivor_activation_not_required")
    if authority.get("candidate_calls_after_registry_write") is not False:
        faults.append("registry_write_improperly_authorizes_candidate_calls")
    if authority.get("serving_training_teacher_D2_or_book_promotion_authority") is not False:
        faults.append("cross_stage_authority_present")
    if authority.get("selected_D1_sources_permanently_excluded_from_training") is not True:
        faults.append("selected_D1_training_exclusion_missing")
    temporal = d1.mapping(config.get("temporal_contamination_guard"))
    if temporal.get(
        "require_task_merged_strictly_after_model_snapshot_observed_utc"
    ) is not True:
        faults.append("post_snapshot_task_requirement_missing")
    if temporal.get("candidate_emitted_contamination_flags_trusted") is not False:
        faults.append("candidate_contamination_flags_trusted")
    return faults


def audit_temporal_guard(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    temporal = d1.mapping(config.get("temporal_contamination_guard"))
    preflight_path = d1.resolve(str(temporal.get("model_runtime_preflight") or ""))
    freeze_path = d1.resolve(str(temporal.get("model_freeze") or ""))
    benchmark_catalog_path = d1.resolve(
        str(temporal.get("public_benchmark_catalog") or "")
    )
    if not preflight_path.is_file():
        faults.append("model_runtime_preflight_missing")
        preflight: dict[str, Any] = {}
    else:
        preflight = d1.read_json(preflight_path)
        if hashlib.sha256(preflight_path.read_bytes()).hexdigest() != temporal.get(
            "model_runtime_preflight_sha256"
        ):
            faults.append("model_runtime_preflight_digest_mismatch")
    if not freeze_path.is_file():
        faults.append("model_freeze_missing")
        freeze: dict[str, Any] = {}
    else:
        freeze = d1.read_json(freeze_path)
        if hashlib.sha256(freeze_path.read_bytes()).hexdigest() != temporal.get(
            "model_freeze_sha256"
        ):
            faults.append("model_freeze_digest_mismatch")
    preflight_identity = d1.mapping(preflight.get("model_identity"))
    freeze_identity_field = str(
        temporal.get("model_freeze_identity_field") or "selected_model_identity"
    )
    freeze_identity = d1.mapping(freeze.get(freeze_identity_field))
    identity_keys = ("repo_id", "revision", "snapshot_manifest_sha256")
    if (
        preflight.get("trigger_state") != "GREEN"
        or any(
            preflight_identity.get(key) != freeze_identity.get(key)
            for key in identity_keys
        )
    ):
        faults.append("frozen_model_identity_or_preflight_invalid")
    observed = str(preflight.get("created_utc") or "")
    if parse_utc(observed) is None:
        faults.append("model_snapshot_observed_utc_invalid")
    benchmark_repositories: list[str] = []
    if not benchmark_catalog_path.is_file():
        faults.append("public_benchmark_catalog_missing")
    else:
        if hashlib.sha256(benchmark_catalog_path.read_bytes()).hexdigest() != temporal.get(
            "public_benchmark_catalog_sha256"
        ):
            faults.append("public_benchmark_catalog_digest_mismatch")
        benchmark_repositories = public_benchmark_repositories(
            d1.read_json(benchmark_catalog_path)
        )
        if not benchmark_repositories:
            faults.append("public_benchmark_repository_inventory_empty")
    return {
        "passed": not faults,
        "faults": sorted(set(faults)),
        "model_runtime_preflight": (
            d1.artifact(preflight_path) if preflight_path.is_file() else {}
        ),
        "model_freeze": d1.artifact(freeze_path) if freeze_path.is_file() else {},
        "model_identity": {
            key: preflight_identity.get(key) for key in identity_keys
        },
        "model_snapshot_observed_utc": observed,
        "public_benchmark_catalog": (
            d1.artifact(benchmark_catalog_path)
            if benchmark_catalog_path.is_file()
            else {}
        ),
        "public_benchmark_repository_count": len(benchmark_repositories),
        "public_benchmark_repositories_sha256": d1.stable_hash(
            benchmark_repositories
        ),
        "public_benchmark_repositories": benchmark_repositories,
        "candidate_emitted_contamination_flags_trusted": False,
    }, faults


def audit_ledger(
    ledger: dict[str, Any],
    config: dict[str, Any],
    *,
    disposition: dict[str, Any],
    disposition_sha256: str,
) -> list[str]:
    faults: list[str] = []
    if ledger.get("policy") != LEDGER_POLICY:
        faults.append("metadata_ledger_policy_invalid")
    if ledger.get("state") != "COMPLETE_QUERY_PARTITIONS_SEALED":
        faults.append("metadata_frame_not_complete_and_sealed")
    if ledger.get("activation_disposition_sha256") != disposition_sha256:
        faults.append("metadata_frame_activation_disposition_mismatch")
    disposition_created = parse_utc(str(disposition.get("created_utc") or ""))
    acquisition_opened = parse_utc(str(ledger.get("acquisition_opened_utc") or ""))
    if (
        disposition_created is None
        or acquisition_opened is None
        or acquisition_opened < disposition_created
    ):
        faults.append("metadata_acquisition_not_proven_post_survivor")
    partitions = dictionaries(ledger.get("query_partitions"))
    if not partitions or any(row.get("complete") is not True for row in partitions):
        faults.append("metadata_query_partition_incomplete")
    if d1.mapping(config.get("discovery_frame")).get("require_raw_response_digests") is True:
        if any(not re.fullmatch(r"[0-9a-f]{64}", str(row.get("raw_response_sha256") or "")) for row in partitions):
            faults.append("metadata_query_partition_digest_invalid")
    boundaries = d1.mapping(ledger.get("boundaries"))
    for key in (
        "external_inference_calls",
        "teacher_calls",
        "candidate_or_control_calls",
        "parent_target_oracle_or_evaluator_executions",
        "archive_fetches",
    ):
        if int(boundaries.get(key) or 0) != 0:
            faults.append(f"metadata_ledger_boundary_nonzero:{key}")
    if contains_forbidden_outcome_key(dictionaries(ledger.get("rows"))):
        faults.append("candidate_control_or_evaluator_outcome_in_metadata_frame")
    for index, row in enumerate(dictionaries(ledger.get("rows"))):
        retrieved = parse_utc(str(row.get("metadata_retrieved_utc") or ""))
        if disposition_created is None or retrieved is None or retrieved < disposition_created:
            faults.append(f"metadata_row_not_retrieved_post_survivor:{index}")
    return faults


def select_rows(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    campaign_id: str,
    prior_repositories: set[str],
    cohort_size: int,
    model_snapshot_observed_utc: str,
    public_benchmark_repositories: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    required = set(d1.mapping(config.get("metadata_eligibility")).get("required_fields") or [])
    instrument = d1.read_json(d1.resolve(str(config["instrument"])))
    allowed_licenses = {
        str(value).lower()
        for value in d1.mapping(instrument.get("source_surface")).get(
            "allowed_license_spdx"
        )
        or []
    }
    allowed_languages = normalized_languages(
        d1.mapping(instrument.get("source_surface")).get(
            "programming_language_scope"
        )
        or []
    )
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        reason = eligibility_fault(
            row,
            required=required,
            allowed_licenses=allowed_licenses,
            allowed_languages=allowed_languages,
            prior_repositories=prior_repositories,
            model_snapshot_observed_utc=model_snapshot_observed_utc,
            public_benchmark_repositories=public_benchmark_repositories,
        )
        if reason:
            exclusions.append({"row": str(index), "reason": reason})
            continue
        normalized = normalize_row(row)
        normalized["selection_digest"] = selection_digest(campaign_id, normalized)
        eligible.append(normalized)
    eligible.sort(key=lambda row: (row["selection_digest"], row["repository"].lower()))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in eligible:
        repository = row["repository"].lower()
        if repository in seen:
            exclusions.append({"row": repository, "reason": "duplicate_repository"})
            continue
        seen.add(repository)
        selected.append(row)
        if len(selected) == cohort_size:
            break
    return selected, exclusions


def eligibility_fault(
    row: dict[str, Any],
    *,
    required: set[str],
    allowed_licenses: set[str],
    allowed_languages: set[str],
    prior_repositories: set[str],
    model_snapshot_observed_utc: str,
    public_benchmark_repositories: set[str],
) -> str:
    if required.difference(row):
        return "required_metadata_missing"
    repository = str(row.get("repository") or "")
    if not REPOSITORY.fullmatch(repository):
        return "repository_identity_invalid"
    if repository.lower() in prior_repositories:
        return "prior_source_repository_overlap"
    if repository.lower() in public_benchmark_repositories:
        return "public_benchmark_repository_overlap"
    if str(row.get("license_spdx") or "").lower() not in allowed_licenses:
        return "license_not_allowlisted"
    if str(row.get("primary_language") or "").lower() not in allowed_languages:
        return "language_out_of_scope"
    revisions = [str(row.get(key) or "").lower() for key in (
        "parent_revision", "target_revision", "merge_revision"
    )]
    if any(not SHA40.fullmatch(value) for value in revisions):
        return "immutable_revision_invalid"
    if revisions[0] == revisions[1]:
        return "parent_target_identity_equal"
    merged = parse_utc(str(row.get("merged_utc") or ""))
    snapshot_observed = parse_utc(model_snapshot_observed_utc)
    if merged is None or snapshot_observed is None or merged <= snapshot_observed:
        return "task_not_merged_strictly_after_frozen_model_snapshot_observation"
    pull = row.get("pull_request")
    if not isinstance(pull, int) or isinstance(pull, bool) or pull < 1:
        return "pull_request_identity_invalid"
    repository_url = f"https://github.com/{repository}"
    if str(row.get("repository_url") or "").rstrip("/") != repository_url:
        return "repository_url_identity_mismatch"
    if str(row.get("pull_request_url") or "") != f"{repository_url}/pull/{pull}":
        return "pull_request_url_identity_mismatch"
    paths = [str(value) for value in row.get("changed_paths") or [] if isinstance(value, str)]
    if not any(path.lower().endswith(SOURCE_SUFFIXES) for path in paths):
        return "changed_in_scope_source_path_missing"
    if not any(any(marker in part.lower() for marker in TEST_MARKERS) for path in paths for part in Path(path).parts):
        return "changed_test_path_missing"
    if row.get("metadata_only_selection") is not True:
        return "metadata_only_selection_not_explicit"
    if contains_forbidden_outcome_key(row):
        return "forbidden_outcome_metadata_present"
    return ""


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "repository",
        "repository_url",
        "license_spdx",
        "primary_language",
        "pull_request",
        "pull_request_url",
        "pull_request_title",
        "merged_utc",
        "parent_revision",
        "target_revision",
        "merge_revision",
        "changed_paths",
        "metadata_retrieved_utc",
    )
    value = {key: row[key] for key in keys}
    value["repository"] = str(value["repository"])
    value["parent_revision"] = str(value["parent_revision"]).lower()
    value["target_revision"] = str(value["target_revision"]).lower()
    value["merge_revision"] = str(value["merge_revision"]).lower()
    value["changed_paths"] = sorted({str(path) for path in value["changed_paths"]})
    return value


def selection_digest(campaign_id: str, row: dict[str, Any]) -> str:
    identity = "||".join((
        campaign_id,
        str(row["repository"]).lower(),
        str(row["parent_revision"]).lower(),
        str(row["target_revision"]).lower(),
    ))
    return hashlib.sha256(identity.encode()).hexdigest()


def contains_forbidden_outcome_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in FORBIDDEN_OUTCOME_KEYS
            or contains_forbidden_outcome_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(contains_forbidden_outcome_key(item) for item in value)
    return False


def normalized_languages(values: Any) -> set[str]:
    normalized: set[str] = set()
    for value in values if isinstance(values, list) else []:
        name = str(value).strip().lower()
        normalized.add(name)
        if name == "javascript/typescript":
            normalized.update({"javascript", "typescript"})
        if name == "html/css":
            normalized.update({"html", "css"})
    return normalized


def public_benchmark_repositories(catalog: dict[str, Any]) -> list[str]:
    repositories: set[str] = set()
    for row in dictionaries(catalog.get("sources")):
        category = str(row.get("category") or "").lower()
        if "benchmark" not in category and "leaderboard" not in category:
            continue
        url = str(row.get("url") or "").rstrip("/")
        match = re.fullmatch(
            r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", url
        )
        if match:
            repositories.add(match.group(1).lower())
    return sorted(repositories)


def parse_utc(value: str) -> datetime | None:
    if not value.endswith("Z"):
        return None
    try:
        return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return None


def source_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "repository": row["repository"],
        "parent_revision": row["parent_revision"],
        "target_revision": row["target_revision"],
        "selection_digest": row["selection_digest"],
    }


def metadata_frame_identity(
    ledger: dict[str, Any], ledger_path: Path, override: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "path": d1.relative(ledger_path) if override is None else "TEST_OVERRIDE_NOT_WRITTEN",
        "sha256": d1.stable_hash(ledger) if override is not None else hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "policy": ledger.get("policy"),
        "state": ledger.get("state"),
        "frame_start_utc": ledger.get("frame_start_utc"),
        "frame_end_utc": ledger.get("frame_end_utc"),
        "row_count": len(dictionaries(ledger.get("rows"))),
        "query_partition_count": len(dictionaries(ledger.get("query_partitions"))),
    }


def reason_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = row["reason"]
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def dictionaries(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "activation_state": report.get("activation_state"),
        "source_metadata_ledger_ingested": report.get(
            "source_metadata_ledger_ingested"
        ),
        "registry_ready": report.get("registry_ready"),
        "candidate_or_control_calls_authorized": report.get(
            "candidate_or_control_calls_authorized"
        ),
        "faults": report.get("faults"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
