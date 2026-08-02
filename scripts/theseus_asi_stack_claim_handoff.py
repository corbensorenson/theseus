#!/usr/bin/env python3
"""Build a public-safe claim-evidence handoff without moving book support."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_asi_stack_claim_handoff.json"
POLICY = "project_theseus_asi_stack_claim_evidence_handoff_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = build_report(config_path)
    out = resolve(args.out or str(config["report"]))
    write_json(out, report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def build_report(
    config_path: Path = DEFAULT_CONFIG,
    *,
    p4_override: dict[str, Any] | None = None,
    d1_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = read_json(config_path)
    faults = validate_config(config)
    book_binding, book_faults = audit_book_binding(config)
    faults.extend(book_faults)
    activation = mapping(config.get("activation"))
    p4_path = resolve(str(activation.get("p4_terminal_disposition") or ""))
    p4 = load_optional(p4_path, p4_override)
    p4_status = str(p4.get("scientific_status") or "")
    p4_green = (
        p4.get("policy") == activation.get("required_p4_policy")
        and p4.get("trigger_state") == activation.get("required_p4_trigger_state")
    )
    survivor = p4_green and p4_status == activation.get("p4_survivor_status")
    non_survivor_terminal = p4_green and p4_status in string_set(
        activation.get("p4_terminal_non_survivor_statuses")
    )

    d1_path = resolve(str(activation.get("d1_terminal_disposition") or ""))
    d1 = load_optional(d1_path, d1_override) if survivor else {}
    d1_terminal = (
        survivor
        and d1.get("policy") == activation.get("required_d1_policy")
        and d1.get("trigger_state") == activation.get("required_d1_trigger_state")
        and str(d1.get("scientific_status") or "")
        in string_set(activation.get("d1_terminal_statuses"))
        and d1.get("claim_id") == mapping(config.get("claim")).get("claim_id")
    )
    packet_ready = not faults and (non_survivor_terminal or d1_terminal)
    if faults:
        trigger_state = "RED"
        activation_state = "BOOK_PIN_CLAIM_OR_HANDOFF_CONTRACT_INVALID"
    elif packet_ready:
        trigger_state = "GREEN"
        activation_state = (
            "READY_FOR_GOVERNED_BOOK_REVIEW_AFTER_D1"
            if survivor
            else "READY_FOR_GOVERNED_BOOK_REVIEW_WITHOUT_D1"
        )
    elif survivor:
        trigger_state = "PAUSED"
        activation_state = "WAITING_FOR_ONE_FRESH_D1_TERMINAL_RESULT"
    else:
        trigger_state = "PAUSED"
        activation_state = "WAITING_FOR_TERMINAL_P4V2R2R3_DISPOSITION"

    report = {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": trigger_state,
        "activation_state": activation_state,
        "faults": sorted(set(faults)),
        "packet_ready": packet_ready,
        "config": artifact(config_path),
        "book_binding": book_binding,
        "p4_terminal_disposition": input_identity(p4_path, p4, p4_override),
        "d1_terminal_disposition": (
            input_identity(d1_path, d1, d1_override) if survivor else {}
        ),
        "book_review_packet": (
            build_packet(config, book_binding=book_binding, p4=p4, d1=d1)
            if packet_ready
            else {}
        ),
        "authority": mapping(config.get("packet_policy")),
        "consumer": mapping(config.get("consumer")),
        "support_state_effect": "none",
        "publication_authority": "none",
        "release_authority": "none",
        "maximum_inference": str(config.get("maximum_inference") or ""),
    }
    if contains_forbidden_packet_key(report["book_review_packet"]):
        report["faults"].append("public_safe_packet_contains_forbidden_key")
        report["trigger_state"] = "RED"
        report["packet_ready"] = False
        report["book_review_packet"] = {}
    return report


def validate_config(config: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    if config.get("state") != "PROSPECTIVELY_BOUND_BEFORE_P4V2R2R3_TERMINAL_EVIDENCE":
        faults.append("state_invalid")
    claim = mapping(config.get("claim"))
    if claim.get("claim_id") != "cognitive-compilation-and-semantic-ir.core":
        faults.append("claim_identity_invalid")
    if claim.get("current_support_state") != "argument":
        faults.append("current_support_state_invalid")
    packet = mapping(config.get("packet_policy"))
    required_true = (
        "public_safe_aggregate_only",
        "negative_results_and_weak_tails_required",
        "exact_artifact_digests_required",
        "maximum_inference_required",
        "book_review_required",
    )
    required_false = (
        "candidate_outputs_included",
        "hidden_tests_or_oracles_included",
        "private_payloads_included",
        "task_selection_metadata_included",
        "automatic_support_transition_proposed",
        "user_or_operator_approval_required",
    )
    if any(packet.get(key) is not True for key in required_true):
        faults.append("required_packet_boundary_missing")
    if any(packet.get(key) is not False for key in required_false):
        faults.append("forbidden_packet_authority_present")
    consumer = mapping(config.get("consumer"))
    if consumer.get("support_state_effect_from_handoff") != "none":
        faults.append("handoff_support_state_effect_present")
    if consumer.get("publication_authority") != "none":
        faults.append("publication_authority_present")
    if consumer.get("release_authority") != "none":
        faults.append("release_authority_present")
    return faults


def audit_book_binding(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    repository = resolve(str(config.get("book_repository") or ""))
    pin = mapping(config.get("book_pin"))
    commit = str(pin.get("commit") or "")
    blobs: dict[str, bytes] = {}
    for key in (
        "manifest_path",
        "claim_matrix_path",
        "evidence_transition_schema_path",
    ):
        path = str(pin.get(key) or "")
        try:
            blobs[key] = git_blob(repository, commit, path)
        except (OSError, subprocess.CalledProcessError):
            faults.append(f"book_pin_blob_unavailable:{key}")
            blobs[key] = b""
    expected_hashes = {
        "manifest_path": "manifest_sha256",
        "claim_matrix_path": "claim_matrix_sha256",
        "evidence_transition_schema_path": "evidence_transition_schema_sha256",
    }
    identities = {}
    for path_key, hash_key in expected_hashes.items():
        observed = sha256_bytes(blobs[path_key]) if blobs[path_key] else ""
        expected = str(pin.get(hash_key) or "")
        passed = len(expected) == 64 and observed == expected
        if not passed:
            faults.append(f"book_pin_digest_mismatch:{path_key}")
        identities[path_key] = {
            "path": pin.get(path_key),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "passed": passed,
        }
    chapter = {}
    if blobs["manifest_path"]:
        try:
            manifest = json.loads(blobs["manifest_path"])
            chapters = collect_chapters(manifest)
            if len(chapters) != int(pin.get("chapter_count") or 0):
                faults.append("book_pin_chapter_count_mismatch")
            claim = mapping(config.get("claim"))
            matches = [row for row in chapters if row.get("file") == claim.get("chapter_path")]
            if len(matches) != 1:
                faults.append("book_claim_chapter_not_unique")
            else:
                chapter = matches[0]
                contract = {
                    key: chapter.get(key)
                    for key in ("id", "chapter_id", "title", "file", "claim_label", "core_claim")
                }
                if stable_hash(contract) != claim.get("claim_contract_sha256"):
                    faults.append("book_claim_contract_mismatch")
                if chapter.get("id") != claim.get("chapter_id"):
                    faults.append("book_chapter_id_mismatch")
                if chapter.get("title") != claim.get("chapter_title"):
                    faults.append("book_chapter_title_mismatch")
        except (json.JSONDecodeError, TypeError, ValueError):
            faults.append("book_manifest_invalid")
    matrix_text = blobs["claim_matrix_path"].decode("utf-8", errors="replace")
    claim = mapping(config.get("claim"))
    expected_row_prefix = (
        f"| `{claim.get('claim_id')}` | `{claim.get('chapter_id')}` |"
    )
    matching_rows = [line for line in matrix_text.splitlines() if line.startswith(expected_row_prefix)]
    if len(matching_rows) != 1:
        faults.append("book_claim_matrix_row_not_unique")
    elif (
        f"| {claim.get('claim_label')} | {claim.get('current_support_state')} |"
        not in matching_rows[0]
    ):
        faults.append("book_claim_matrix_state_mismatch")
    return {
        "passed": not faults,
        "repository": relative(repository),
        "commit": commit,
        "chapter_count": int(pin.get("chapter_count") or 0),
        "claim_id": claim.get("claim_id"),
        "chapter_id": claim.get("chapter_id"),
        "current_support_state": claim.get("current_support_state"),
        "claim_contract_sha256": claim.get("claim_contract_sha256"),
        "artifacts": identities,
    }, faults


def build_packet(
    config: dict[str, Any],
    *,
    book_binding: dict[str, Any],
    p4: dict[str, Any],
    d1: dict[str, Any],
) -> dict[str, Any]:
    claim = mapping(config.get("claim"))
    p4_status = str(p4.get("scientific_status") or "")
    survivor = p4_status == mapping(config.get("activation")).get("p4_survivor_status")
    return {
        "packet_policy": "project_theseus_public_safe_claim_evidence_packet_v1",
        "claim_id": claim.get("claim_id"),
        "chapter_id": claim.get("chapter_id"),
        "book_pin_commit": book_binding.get("commit"),
        "book_claim_contract_sha256": claim.get("claim_contract_sha256"),
        "current_book_support_state": claim.get("current_support_state"),
        "evidence_route": "P4V2R2R3_complete_artifact_then_one_fresh_D1_only_if_survivor",
        "p4": public_terminal_summary(p4),
        "d1": public_terminal_summary(d1) if survivor else {},
        "negative_results": public_strings(p4.get("negative_results"))
        + public_strings(d1.get("negative_results")),
        "weak_tail_summary": {
            "p4": select_keys(mapping(p4.get("weak_tail")), ("minimum", "maximum", "median", "count")),
            "d1": select_keys(mapping(d1.get("weak_tail")), ("minimum", "maximum", "median", "count")),
        },
        "review_request": {
            "kind": "claim_level_evidence_review",
            "automatic_transition_proposed": False,
            "support_state_effect": "none",
            "book_must_create_separate_evidence_transition_record": True,
            "book_review_may_accept_reject_narrow_or_block": True,
        },
        "limitations": [
            "The evidence is scoped to the exact frozen local model, mechanism implementation, task surfaces, controls, evaluator, budgets, and operating regime.",
            "P4 is decision-development evidence; a survivor requires one fresh source-disjoint D1 result before handoff.",
            "A non-survivor or inconclusive result does not broadly falsify cognitive compilation.",
            "The handoff does not establish cross-language transfer, serving or training readiness, AGI, ASI, or book support movement.",
        ],
        "support_state_effect": "none",
        "publication_authority": "none",
        "release_authority": "none",
        "maximum_inference": str(config.get("maximum_inference") or ""),
    }


def public_terminal_summary(value: dict[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    return {
        "policy": value.get("policy"),
        "claim_id": value.get("claim_id"),
        "trigger_state": value.get("trigger_state"),
        "scientific_status": value.get("scientific_status"),
        "scope": value.get("scope"),
        "denominators": select_keys(
            mapping(value.get("denominators")),
            (
                "tasks",
                "learned_model_calls",
                "hosted_model_calls",
                "teacher_calls",
                "project_selected_quality_token_cap",
            ),
        ),
        "adequacy": select_keys(
            mapping(value.get("adequacy")),
            (
                "information_flow_green",
                "mechanics_floor_passed",
                "experiment_floor_passed",
                "independent_evaluator_replay_green",
            ),
        ),
        "decision_rule": select_keys(
            mapping(value.get("decision_rule")),
            (
                "survivor_effect_rule_passed",
                "effect_decision_authorized",
                "primary_effect",
                "confidence_interval",
            ),
        ),
        "consumption": select_keys(
            mapping(value.get("consumption")),
            ("eligible_for_D1", "exact_surface_consumed", "rerun_allowed"),
        ),
        "maximum_inference": value.get("maximum_inference"),
        "report_sha256": stable_hash(value),
    }


FORBIDDEN_PACKET_KEYS = {
    "answer",
    "candidate_output",
    "control_output",
    "expected",
    "hidden_tests",
    "oracle",
    "solution",
    "solution_body",
    "solution_expr",
    "source_task_id",
    "tests",
}


def contains_forbidden_packet_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in FORBIDDEN_PACKET_KEYS
            or contains_forbidden_packet_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(contains_forbidden_packet_key(item) for item in value)
    return False


def git_blob(repository: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{commit}:{path}"],
        stderr=subprocess.DEVNULL,
        timeout=20,
    )


def collect_chapters(value: Any) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if isinstance(value.get("file"), str) and str(value["file"]).startswith("chapters/"):
            chapters.append(value)
        for item in value.values():
            chapters.extend(collect_chapters(item))
    elif isinstance(value, list):
        for item in value:
            chapters.extend(collect_chapters(item))
    return chapters


def input_identity(path: Path, value: dict[str, Any], override: Any) -> dict[str, Any]:
    if not value:
        return {}
    return {
        "path": "TEST_OVERRIDE_NOT_ON_DISK" if override is not None else relative(path),
        "sha256": stable_hash(value) if override is not None else sha256_bytes(path.read_bytes()),
        "policy": value.get("policy"),
        "trigger_state": value.get("trigger_state"),
        "scientific_status": value.get("scientific_status"),
    }


def load_optional(path: Path, override: dict[str, Any] | None) -> dict[str, Any]:
    if override is not None:
        return override
    return read_json(path) if path.is_file() else {}


def select_keys(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: value[key] for key in keys if key in value}


def public_strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def string_set(value: Any) -> set[str]:
    return {str(item) for item in value} if isinstance(value, list) else set()


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact(path: Path) -> dict[str, Any]:
    return {"path": relative(path), "sha256": sha256_bytes(path.read_bytes())}


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "activation_state": report.get("activation_state"),
        "book_binding_passed": mapping(report.get("book_binding")).get("passed"),
        "packet_ready": report.get("packet_ready"),
        "support_state_effect": report.get("support_state_effect"),
        "faults": report.get("faults"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
