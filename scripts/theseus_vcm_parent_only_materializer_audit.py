#!/usr/bin/env python3
"""Role-separated audit of the K2.04 parent-only store/materializer boundary."""
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

POLICY = "project_theseus_vcm_parent_only_materializer_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_parent_only_materializer.json"
ROW_KEYS = {
    "natural_language_request", "natural_language_request_sha256", "parent_archive",
    "parent_archive_sha256", "parent_archive_root", "parent_revision", "license_spdx",
    "sanitization_report", "sanitization_report_sha256",
}
VISIBLE_KEYS = {
    "natural_language_request", "callable_signature_when_present",
    "broad_parent_effect_root", "arm_specific_model_visible_context",
}
FORBIDDEN_KEYS = {
    "allowed_effect_paths", "answer", "answer_family", "category", "expected",
    "hidden_tests", "repository_identity", "required_constructs", "return_shape",
    "solution", "solution_body", "solution_expr", "source_task_id", "target_diff",
    "target_patch", "target_snapshot", "tests", "type_family",
}
TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOP_WORDS = frozenset({"and", "are", "but", "close", "feat", "fix", "for", "from", "into", "not", "per", "that", "the", "this", "was", "with"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report = audit(path)
    p2a.write_json(p2a.resolve(args.out or str(cfg["audit_report"])), report)
    print(json.dumps({key: report.get(key) for key in ("trigger_state", "state", "faults", "audited_row_count", "audited_candidate_visible_field_count", "candidate_or_control_calls", "external_reference_calls")}, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(path: Path = DEFAULT_CONFIG, *, producer: dict[str, Any] | None = None, store: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    audit_owner = p2a.resolve(str(cfg.get("audit_owner") or ""))
    producer_owner = p2a.resolve(str(cfg.get("owner") or ""))
    if cfg.get("audit_policy") != POLICY:
        faults.append("audit_policy_invalid")
    if audit_owner != Path(__file__).resolve() or p2a.sha256_file(audit_owner) != cfg.get("audit_owner_sha256"):
        faults.append("audit_owner_binding_invalid")
    if not producer_owner.is_file() or p2a.sha256_file(producer_owner) != cfg.get("owner_sha256"):
        faults.append("producer_owner_binding_invalid")
    producer_report = producer if producer is not None else p2a.read_json(p2a.resolve(str(cfg.get("report") or "")))
    store_payload = store if store is not None else p2a.read_json(p2a.resolve(str(cfg.get("store_out") or "")))
    if producer_report.get("trigger_state") != "GREEN":
        faults.append("producer_not_green")
    if store_payload.get("policy") != "project_theseus_vcm_parent_archive_store_v1":
        faults.append("store_policy_invalid")
    if producer_report.get("candidate_or_control_calls") != 0 or producer_report.get("external_reference_calls") != 0:
        faults.append("producer_downstream_counter_invalid")

    producer_rows = {str(row.get("request_id")): row for row in p2a.dicts(producer_report.get("rows"))}
    store_rows = {str(row.get("request_id")): row for row in p2a.dicts(store_payload.get("rows"))}
    audit_rows: list[dict[str, Any]] = []
    visible_count = 0
    for binding in p2a.dicts(cfg.get("rows")):
        if set(binding) != ROW_KEYS:
            faults.append("row_binding_field_set_invalid")
        if any(key in FORBIDDEN_KEYS for key in binding):
            faults.append("forbidden_row_binding_field")
        request = str(binding.get("natural_language_request") or "")
        if digest(request.encode("utf-8")) != binding.get("natural_language_request_sha256"):
            faults.append("request_digest_invalid")
        archive = p2a.resolve(str(binding.get("parent_archive") or ""))
        sanitation = p2a.resolve(str(binding.get("sanitization_report") or ""))
        if not archive.name.endswith("_parent.tar.gz") or "target" in archive.name.lower():
            faults.append("non_parent_archive_bound")
        if not sanitation.name.endswith("_parent.json") or "target" in sanitation.name.lower():
            faults.append("non_parent_sanitization_bound")
        if p2a.sha256_file(archive) != binding.get("parent_archive_sha256"):
            faults.append("archive_digest_invalid")
        if p2a.sha256_file(sanitation) != binding.get("sanitization_report_sha256"):
            faults.append("sanitization_digest_invalid")
        inventory, frontier, scan_faults = independent_scan(archive, str(binding.get("parent_archive_root") or ""), request)
        faults.extend(scan_faults)
        request_id = digest(canonical({"request_sha256": binding.get("natural_language_request_sha256"), "parent_archive_sha256": binding.get("parent_archive_sha256"), "selector_policy": "project_theseus_vcm_request_parent_lexical_frontier_v1"}))
        produced = producer_rows.get(request_id, {})
        stored = store_rows.get(request_id, {})
        if produced.get("inventory_sha256") != digest(canonical(inventory)) or stored.get("inventory_sha256") != digest(canonical(inventory)):
            faults.append(f"inventory_rederivation_failed:{request_id}")
        selector = p2a.mapping(stored.get("selector"))
        if produced.get("selector_frontier_sha256") != digest(canonical(frontier)) or selector.get("frontier_sha256") != digest(canonical(frontier)):
            faults.append(f"selector_frontier_rederivation_failed:{request_id}")
        if selector.get("frontier") != frontier:
            faults.append(f"selector_frontier_content_invalid:{request_id}")
        if selector.get("selected_page_or_byte_cap") is not None or selector.get("frontier_is_complete_for_all_utf8_text_pages") is not True:
            faults.append(f"selector_convenience_cap_or_omission:{request_id}")
        surface = p2a.mapping(produced.get("candidate_surface"))
        if set(surface) != VISIBLE_KEYS:
            faults.append(f"candidate_surface_field_set_invalid:{request_id}")
        forbidden = recursive_forbidden_keys(surface)
        if forbidden:
            faults.append(f"candidate_surface_forbidden_fields:{request_id}:{','.join(forbidden)}")
        if surface.get("broad_parent_effect_root") != "repository" or produced.get("allowed_effect_paths_present") is not False:
            faults.append(f"broad_effect_boundary_invalid:{request_id}")
        if surface.get("natural_language_request") != request:
            faults.append(f"candidate_request_mismatch:{request_id}")
        receipts = {str(row.get("field")): row for row in p2a.dicts(produced.get("candidate_visible_byte_receipts"))}
        for key, value in surface.items():
            payload = canonical(value)
            receipt = receipts.get(key, {})
            if receipt.get("utf8_bytes") != len(payload) or receipt.get("sha256") != digest(payload):
                faults.append(f"candidate_visible_byte_receipt_invalid:{request_id}:{key}")
            visible_count += 1
        if produced.get("candidate_visible_projection_sha256") != digest(canonical(surface)):
            faults.append(f"candidate_visible_projection_invalid:{request_id}")
        matched = p2a.mapping(produced.get("matched_context_materialization_receipts"))
        vcm_set = p2a.mapping(matched.get("governed_vcm"))
        for route_id in (
            "information_matched_plain_context",
            "maximal_full_parent_context",
            "ordinary_direct_retrieval",
        ):
            route_set = p2a.mapping(matched.get(route_id))
            if (
                route_set.get("information_set_sha256") != vcm_set.get("information_set_sha256")
                or route_set.get("parent_text_page_count") != vcm_set.get("parent_text_page_count")
                or route_set.get("parent_text_bytes") != vcm_set.get("parent_text_bytes")
                or route_set.get("project_selected_page_or_byte_cap") is not None
            ):
                faults.append(f"matched_context_information_mismatch:{request_id}:{route_id}")
        no_context = p2a.mapping(matched.get("no_added_context"))
        if no_context.get("parent_text_page_count") != 0 or no_context.get("parent_text_bytes") != 0:
            faults.append(f"no_added_context_not_empty:{request_id}")
        if produced.get("matched_context_information_identity_preserved") is not True:
            faults.append(f"matched_context_identity_receipt_missing:{request_id}")
        abi = p2a.mapping(produced.get("vcm_consumer_abi"))
        if abi.get("ready") is not True or abi.get("validation", {}).get("passed") is not True:
            faults.append(f"production_vcm_abi_not_ready:{request_id}")
        audit_rows.append({
            "request_id": request_id,
            "parent_archive_sha256": binding.get("parent_archive_sha256"),
            "regular_file_count": len(inventory),
            "text_page_count": len(frontier),
            "inventory_sha256": digest(canonical(inventory)),
            "selector_frontier_sha256": digest(canonical(frontier)),
            "candidate_visible_projection_sha256": digest(canonical(surface)),
            "candidate_visible_field_count": len(surface),
            "broad_parent_effect_root": surface.get("broad_parent_effect_root"),
            "vcm_consumer_abi_ready": abi.get("ready") is True,
        })

    if len(audit_rows) != int(cfg.get("expected_row_count") or 0):
        faults.append("audited_row_count_invalid")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "K2_04_PARENT_ONLY_MATERIALIZER_ROLE_SEPARATELY_REDERIVED" if not faults else "K2_04_PARENT_ONLY_MATERIALIZER_AUDIT_FAILED",
        "faults": sorted(set(faults)),
        "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)},
        "audited_row_count": len(audit_rows),
        "audited_candidate_visible_field_count": visible_count,
        "rows": audit_rows,
        "audit_kind": "role-separated rederivation",
        "conclusions": {
            "exact_parent_archives_only": not faults,
            "complete_parent_text_frontier_no_convenience_cap": not faults,
            "selector_inputs_request_and_parent_only": not faults,
            "candidate_visible_bytes_individually_rederived": not faults,
            "single_broad_parent_effect_root": not faults,
            "target_derived_effect_paths_absent": not faults,
            "matched_non_vcm_context_information_identity": not faults,
            "production_vcm_consumer_abi_ready": not faults,
        },
        "network_or_external_execution_performed": False,
        "repository_runner_executions": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": cfg.get("audit_maximum_inference"),
    }


def independent_scan(archive: Path, expected_root: str, request: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    faults: list[str] = []
    inventory: list[dict[str, Any]] = []
    texts: list[tuple[dict[str, Any], str]] = []
    try:
        with tarfile.open(archive, "r:gz") as handle:
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
                extracted = handle.extractfile(member)
                if not relative or extracted is None:
                    faults.append("archive_regular_member_invalid")
                    continue
                payload = extracted.read()
                text = independent_decode(payload)
                row = {"path": relative, "member": member.name, "bytes": len(payload), "sha256": digest(payload), "content_class": "utf8_text" if text is not None else "opaque_binary"}
                inventory.append(row)
                if text is not None:
                    texts.append((row, text))
    except (tarfile.TarError, OSError) as exc:
        faults.append(f"parent_archive_read_failed:{type(exc).__name__}")
    inventory.sort(key=lambda row: row["path"])
    terms = sorted({token for token in TOKEN_RE.findall(request.lower()) if len(token) >= 3 and token not in STOP_WORDS})
    frontier = []
    for row, text in texts:
        path_text, content_text = row["path"].lower(), text.lower()
        path_hits = sum(path_text.count(term) for term in terms)
        content_hits = sum(min(content_text.count(term), 32) for term in terms)
        frontier.append({"path": row["path"], "member": row["member"], "bytes": row["bytes"], "sha256": row["sha256"], "score": path_hits * 64 + content_hits, "path_term_hits": path_hits, "content_term_hits": content_hits})
    frontier.sort(key=lambda row: (-int(row["score"]), str(row["path"])))
    return inventory, frontier, sorted(set(faults))


def independent_decode(payload: bytes) -> str | None:
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


def recursive_forbidden_keys(value: Any) -> list[str]:
    hits: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in FORBIDDEN_KEYS:
                hits.add(str(key))
            hits.update(recursive_forbidden_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            hits.update(recursive_forbidden_keys(nested))
    return sorted(hits)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
