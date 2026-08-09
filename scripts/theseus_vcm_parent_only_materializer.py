#!/usr/bin/env python3
"""Build an archive-backed, parent-only VCM store and governed request packets.

This owner deliberately does not extract or inspect a target snapshot.  Every
retrieval coordinate is derived from the natural-language request and the exact
regular-file contents of one immutable parent archive.  The complete text
frontier is retained; context/addressability policy is owned by the later
campaign freeze rather than hidden here as a convenience cutoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import vcm_consumer_abi  # noqa: E402

POLICY = "project_theseus_vcm_parent_only_materializer_v1"
STORE_POLICY = "project_theseus_vcm_parent_archive_store_v1"
SELECTOR_POLICY = "project_theseus_vcm_request_parent_lexical_frontier_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_parent_only_materializer.json"
TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOP_WORDS = frozenset(
    {
        "and", "are", "but", "close", "feat", "fix", "for", "from", "into",
        "not", "per", "that", "the", "this", "was", "with",
    }
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--store-out", default="")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    cfg = p2a.read_json(config_path)
    report, store = materialize(config_path)
    store_path = p2a.resolve(args.store_out or str(cfg["store_out"]))
    report_path = p2a.resolve(args.out or str(cfg["report"]))
    p2a.write_json(store_path, store)
    report["store_artifact"] = {
        "path": p2a.rel(store_path),
        "sha256": p2a.sha256_file(store_path),
        "bytes": store_path.stat().st_size,
    }
    p2a.write_json(report_path, report)
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "trigger_state", "state", "faults", "row_count",
                    "regular_file_count", "text_page_count",
                    "candidate_or_control_calls", "external_reference_calls",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["trigger_state"] == "GREEN" else 2


def materialize(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    owner = p2a.resolve(str(cfg.get("owner") or ""))
    if cfg.get("policy") != POLICY:
        faults.append("policy_invalid")
    if owner != Path(__file__).resolve() or not owner.is_file():
        faults.append("owner_invalid")
    elif p2a.sha256_file(owner) != cfg.get("owner_sha256"):
        faults.append("owner_binding_invalid")
    broad_root = str(cfg.get("broad_parent_effect_root") or "")
    if broad_root != "repository":
        faults.append("broad_parent_effect_root_invalid")

    store_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for binding in p2a.dicts(cfg.get("rows")):
        row, receipt_faults = build_row(binding, path, cfg)
        faults.extend(receipt_faults)
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in seen_ids:
            faults.append("request_id_invalid_or_duplicate")
        seen_ids.add(request_id)
        store_rows.append(row["store_row"])
        report_rows.append(row["report_row"])

    store = {
        "policy": STORE_POLICY,
        "created_utc": p2a.now(),
        "source_boundary": "exact_immutable_parent_archives_only",
        "content_storage": "archive_backed_no_duplicate_payload",
        "selector_policy": SELECTOR_POLICY,
        "rows": store_rows,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "parent_target_or_evaluator_executions": 0,
    }
    ready = not faults and len(report_rows) == int(cfg.get("expected_row_count") or 0)
    if len(report_rows) != int(cfg.get("expected_row_count") or 0):
        faults.append("row_count_invalid")
        ready = False
    return (
        {
            "policy": POLICY,
            "created_utc": p2a.now(),
            "trigger_state": "GREEN" if ready else "RED",
            "state": (
                "K2_04_PARENT_ONLY_STORE_AND_REQUEST_MATERIALIZER_GREEN"
                if ready
                else "K2_04_PARENT_ONLY_MATERIALIZER_FAILED"
            ),
            "faults": sorted(set(faults)),
            "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)},
            "row_count": len(report_rows),
            "regular_file_count": sum(int(row.get("regular_file_count") or 0) for row in report_rows),
            "text_page_count": sum(int(row.get("text_page_count") or 0) for row in report_rows),
            "rows": report_rows,
            "information_flow": {
                "selector_inputs": [
                    "natural_language_request_utf8",
                    "parent_regular_file_relative_path_utf8",
                    "parent_regular_file_content_bytes",
                    "fixed_selector_policy",
                ],
                "candidate_visible_fields": [
                    "natural_language_request",
                    "callable_signature_when_present",
                    "broad_parent_effect_root",
                    "arm_specific_model_visible_context",
                ],
                "target_archive_read": False,
                "target_diff_read": False,
                "target_selected_path_read": False,
                "allowed_effect_paths_present": False,
                "answer_identifying_metadata_read": False,
                "complete_parent_text_frontier_retained_without_selection_cap": True,
            },
            "audit_kind": "producer receipt; separate role audit required",
            "candidate_or_control_calls": 0,
            "external_reference_calls": 0,
            "parent_target_or_evaluator_executions": 0,
            "repository_runner_executions": 0,
            "network_fetches": 0,
            "maximum_inference": cfg.get("maximum_inference"),
        },
        store,
    )


def build_row(binding: dict[str, Any], config_path: Path, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    allowed_keys = {
        "natural_language_request", "natural_language_request_sha256", "parent_archive",
        "parent_archive_sha256", "parent_archive_root", "parent_revision",
        "license_spdx", "sanitization_report", "sanitization_report_sha256",
    }
    if set(binding) != allowed_keys:
        faults.append("row_binding_field_set_invalid")
    request = str(binding.get("natural_language_request") or "")
    if sha256_bytes(request.encode("utf-8")) != binding.get("natural_language_request_sha256"):
        faults.append("natural_language_request_hash_invalid")
    archive_path = p2a.resolve(str(binding.get("parent_archive") or ""))
    if not archive_path.name.endswith("_parent.tar.gz") or "target" in archive_path.name.lower():
        faults.append("parent_archive_name_invalid")
    if p2a.sha256_file(archive_path) != binding.get("parent_archive_sha256"):
        faults.append("parent_archive_hash_invalid")
    sanitation = p2a.resolve(str(binding.get("sanitization_report") or ""))
    if not sanitation.name.endswith("_parent.json") or p2a.sha256_file(sanitation) != binding.get("sanitization_report_sha256"):
        faults.append("parent_sanitization_binding_invalid")

    inventory, frontier, scan_faults = scan_parent_archive(
        archive_path,
        str(binding.get("parent_archive_root") or ""),
        request,
    )
    faults.extend(scan_faults)
    inventory_sha = canonical_sha256(inventory)
    frontier_sha = canonical_sha256(frontier)
    request_id = sha256_bytes(
        canonical_bytes(
            {
                "request_sha256": binding.get("natural_language_request_sha256"),
                "parent_archive_sha256": binding.get("parent_archive_sha256"),
                "selector_policy": SELECTOR_POLICY,
            }
        )
    )
    store_handle = f"vcm://parent/{request_id}/{inventory_sha}"
    store_row = {
        "request_id": request_id,
        "source_snapshot": {
            "archive": p2a.rel(archive_path),
            "archive_sha256": binding.get("parent_archive_sha256"),
            "archive_root": binding.get("parent_archive_root"),
            "revision": binding.get("parent_revision"),
            "license_spdx": binding.get("license_spdx"),
            "sanitization_report": p2a.rel(sanitation),
            "sanitization_report_sha256": binding.get("sanitization_report_sha256"),
        },
        "inventory": inventory,
        "inventory_sha256": inventory_sha,
        "selector": {
            "policy": SELECTOR_POLICY,
            "request_terms": request_terms(request),
            "input_classes": ["request_utf8", "parent_relative_path_utf8", "parent_content_bytes", "fixed_policy"],
            "frontier": frontier,
            "frontier_sha256": frontier_sha,
            "frontier_is_complete_for_all_utf8_text_pages": True,
            "selected_page_or_byte_cap": None,
        },
        "content_resolver": {
            "policy": "exact_tar_member_read_sha256_revalidation_v1",
            "store_handle": store_handle,
            "payload_duplicated": False,
        },
    }
    context_ref = {
        "kind": "artifact_ref",
        "ref": p2a.rel(archive_path),
        "required": True,
        "exists": archive_path.is_file(),
        "sha256": binding.get("parent_archive_sha256"),
        "taint_labels": [],
        "contradiction_refs": [],
    }
    packet = vcm_consumer_abi.build_consumer_packet(
        consumer_id=f"theseus_vcm_parent_only_materializer:{request_id}",
        purpose="parent_only_repository_context_materialization",
        read_set=[p2a.rel(archive_path), p2a.rel(sanitation)],
        write_set=[str(cfg.get("broad_parent_effect_root")), str(cfg.get("store_out")), str(cfg.get("report"))],
        authority_ceiling=["licensed_parent_snapshot_read", "governed_vcm_packet_materialization"],
        permitted_uses=["measurement_candidate_context", "blind_retrieval", "audit_replay"],
        context_refs=[context_ref],
        materialized_authority_labels=["licensed_parent_snapshot_read"],
        deletion_obligations=["invalidate_packet_if_parent_snapshot_or_license_is_revoked"],
        audit_refs=[p2a.rel(config_path), "scripts/theseus_vcm_parent_only_materializer.py"],
    )
    if not packet.get("ready"):
        faults.extend(f"vcm_consumer_abi:{value}" for value in p2a.strings(packet.get("typed_faults")))
    candidate_surface = {
        "natural_language_request": request,
        "callable_signature_when_present": None,
        "broad_parent_effect_root": str(cfg.get("broad_parent_effect_root")),
        "arm_specific_model_visible_context": {
            "vcm_store_handle": store_handle,
            "governed_packet_id": packet.get("packet_id"),
            "materialized_parent_pages": [],
            "materialization_state": "deferred_until_obligation_and_physical_addressability_freeze",
        },
    }
    shared_information = {
        "parent_text_frontier_sha256": frontier_sha,
        "parent_text_page_count": len(frontier),
        "parent_text_bytes": sum(int(row["bytes"]) for row in frontier),
        "context_addressability_boundary": "deferred_to_exact_frozen_tokenizer_and_host_preflight",
        "project_selected_page_or_byte_cap": None,
    }
    matched_contexts = {
        "no_added_context": {
            "information_set_sha256": canonical_sha256([]),
            "parent_text_page_count": 0,
            "parent_text_bytes": 0,
        },
        "governed_vcm": {
            **shared_information,
            "information_set_sha256": frontier_sha,
            "governance_envelope": "production_vcm_consumer_abi",
        },
        "information_matched_plain_context": {
            **shared_information,
            "information_set_sha256": frontier_sha,
            "governance_envelope": None,
        },
        "maximal_full_parent_context": {
            **shared_information,
            "information_set_sha256": frontier_sha,
            "ordering": "stable_request_derived_complete_frontier",
        },
        "ordinary_direct_retrieval": {
            **shared_information,
            "information_set_sha256": frontier_sha,
            "retrieval_frontier": "same_complete_parent_text_frontier",
        },
    }
    visible_receipts = byte_receipts(candidate_surface)
    report_row = {
        "request_id": request_id,
        "parent_archive_sha256": binding.get("parent_archive_sha256"),
        "parent_revision_sha256": sha256_bytes(str(binding.get("parent_revision") or "").encode("utf-8")),
        "license_spdx": binding.get("license_spdx"),
        "regular_file_count": len(inventory),
        "regular_file_bytes": sum(int(row["bytes"]) for row in inventory),
        "text_page_count": len(frontier),
        "text_page_bytes": sum(int(row["bytes"]) for row in frontier),
        "inventory_sha256": inventory_sha,
        "selector_frontier_sha256": frontier_sha,
        "selector_input_classes": store_row["selector"]["input_classes"],
        "selector_has_no_page_or_byte_cap": True,
        "candidate_surface": candidate_surface,
        "candidate_visible_byte_receipts": visible_receipts,
        "candidate_visible_projection_sha256": canonical_sha256(candidate_surface),
        "matched_context_materialization_receipts": matched_contexts,
        "matched_context_information_identity_preserved": True,
        "broad_effect_root_is_common_and_not_target_derived": True,
        "allowed_effect_paths_present": False,
        "vcm_consumer_abi": vcm_consumer_abi.compact_consumer_packet(packet),
        "faults": sorted(set(faults)),
    }
    return {"request_id": request_id, "store_row": store_row, "report_row": report_row}, faults


def scan_parent_archive(archive_path: Path, expected_root: str, request: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    faults: list[str] = []
    inventory: list[dict[str, Any]] = []
    text_rows: list[tuple[dict[str, Any], str]] = []
    if not archive_path.is_file():
        return [], [], ["parent_archive_missing"]
    try:
        with tarfile.open(archive_path, "r:gz") as handle:
            for member in handle.getmembers():
                parts = PurePosixPath(member.name).parts
                if not parts or parts[0] != expected_root or any(part in {"", ".", ".."} for part in parts):
                    faults.append("archive_member_root_or_path_invalid")
                    continue
                if member.isdir():
                    continue
                if not member.isfile():
                    faults.append("archive_non_regular_member_denied")
                    continue
                relative = PurePosixPath(*parts[1:]).as_posix()
                if not relative or relative.startswith("/"):
                    faults.append("archive_relative_path_invalid")
                    continue
                extracted = handle.extractfile(member)
                if extracted is None:
                    faults.append("archive_regular_member_unreadable")
                    continue
                payload = extracted.read()
                if len(payload) != member.size:
                    faults.append("archive_member_size_mismatch")
                text = decode_text(payload)
                row = {
                    "path": relative,
                    "member": member.name,
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                    "content_class": "utf8_text" if text is not None else "opaque_binary",
                }
                inventory.append(row)
                if text is not None:
                    text_rows.append((row, text))
    except (tarfile.TarError, OSError) as exc:
        faults.append(f"parent_archive_read_failed:{type(exc).__name__}")
    inventory.sort(key=lambda row: row["path"])
    terms = request_terms(request)
    frontier = []
    for row, text in text_rows:
        score, path_hits, content_hits = lexical_score(row["path"], text, terms)
        frontier.append(
            {
                "path": row["path"],
                "member": row["member"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "score": score,
                "path_term_hits": path_hits,
                "content_term_hits": content_hits,
            }
        )
    frontier.sort(key=lambda row: (-int(row["score"]), str(row["path"])))
    return inventory, frontier, sorted(set(faults))


def read_parent_page(store_row: dict[str, Any], relative_path: str) -> bytes:
    source = p2a.mapping(store_row.get("source_snapshot"))
    inventory = {str(row.get("path")): row for row in p2a.dicts(store_row.get("inventory"))}
    receipt = inventory.get(relative_path)
    if receipt is None:
        raise KeyError(relative_path)
    archive = p2a.resolve(str(source.get("archive") or ""))
    with tarfile.open(archive, "r:gz") as handle:
        extracted = handle.extractfile(str(receipt["member"]))
        if extracted is None:
            raise ValueError("archive member missing")
        payload = extracted.read()
    if len(payload) != receipt.get("bytes") or sha256_bytes(payload) != receipt.get("sha256"):
        raise ValueError("parent page integrity mismatch")
    return payload


def request_terms(request: str) -> list[str]:
    return sorted({token for token in TOKEN_RE.findall(request.lower()) if len(token) >= 3 and token not in STOP_WORDS})


def lexical_score(path: str, text: str, terms: list[str]) -> tuple[int, int, int]:
    lower_path = path.lower()
    lower_text = text.lower()
    path_hits = sum(lower_path.count(term) for term in terms)
    content_hits = sum(min(lower_text.count(term), 32) for term in terms)
    return path_hits * 64 + content_hits, path_hits, content_hits


def decode_text(payload: bytes) -> str | None:
    if b"\0" in payload:
        return None
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text:
        return ""
    controls = sum(1 for char in text if ord(char) < 32 and char not in "\n\r\t\f\b")
    return None if controls / len(text) > 0.01 else text


def byte_receipts(surface: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for field, value in surface.items():
        payload = canonical_bytes(value)
        rows.append({"field": field, "utf8_bytes": len(payload), "sha256": sha256_bytes(payload)})
    return rows


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
