#!/usr/bin/env python3
"""Call-free VCM v4 repair: reconstructable sub-file pages and matched pairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_k3_route_preflight as v3  # noqa: E402

POLICY = "project_theseus_vcm_v4_chunked_route_preflight_v2"
CONFIG_POLICY = "project_theseus_vcm_v4_chunked_route_preflight_config_v2"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_v4_chunked_route_preflight.json"


def chunk_text(path: str, text: str, maximum_chars: int) -> list[dict[str, Any]]:
    if maximum_chars < 1:
        raise ValueError("maximum_chars must be positive")
    rows: list[dict[str, Any]] = []
    start = 0
    ordinal = 1
    while start < len(text):
        hard_end = min(len(text), start + maximum_chars)
        end = hard_end
        if hard_end < len(text):
            newline = text.rfind("\n", start, hard_end)
            if newline >= start:
                end = newline + 1
        if end <= start:
            end = hard_end
        payload = text[start:end]
        rows.append({
            "path": path,
            "chunk_id": f"{path}#chars={start}:{end}",
            "chunk_ordinal": ordinal,
            "start_char": start,
            "end_char": end,
            "chars": len(payload),
            "bytes": len(payload.encode("utf-8")),
            "sha256": p2a.sha256_text(payload),
            "text": payload,
        })
        start = end
        ordinal += 1
    if not rows:
        rows.append({
            "path": path, "chunk_id": f"{path}#chars=0:0", "chunk_ordinal": 1,
            "start_char": 0, "end_char": 0, "chars": 0, "bytes": 0,
            "sha256": p2a.sha256_text(""), "text": "",
        })
    if "".join(row["text"] for row in rows) != text:
        raise AssertionError(f"chunk reconstruction failed: {path}")
    return rows


def identity(path: Path) -> dict[str, Any]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path), "bytes": path.stat().st_size}


def validate_binding(cfg: dict[str, Any], path_key: str, hash_key: str, faults: list[str]) -> None:
    path = p2a.resolve(str(cfg.get(path_key) or ""))
    if not path.is_file() or p2a.sha256_file(path) != cfg.get(hash_key):
        faults.append(f"binding_invalid:{path_key}")


def read_bound(cfg: dict[str, Any], path_key: str, hash_key: str, faults: list[str]) -> dict[str, Any]:
    validate_binding(cfg, path_key, hash_key, faults)
    try:
        return p2a.read_json(p2a.resolve(str(cfg.get(path_key) or "")))
    except Exception as exc:
        faults.append(f"binding_read_failed:{path_key}:{type(exc).__name__}")
        return {}


def build(
    path: Path = DEFAULT_CONFIG,
    *,
    token_counter: Callable[[str, str], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != CONFIG_POLICY:
        faults.append("config_policy_invalid")
    for key, digest in (
        ("owner", "owner_sha256"), ("audit_owner", "audit_owner_sha256"),
        ("test", "test_sha256"), ("v3_config", "v3_config_sha256"),
        ("v3_report", "v3_report_sha256"), ("v3_audit", "v3_audit_sha256"),
        ("v3_host_report", "v3_host_report_sha256"),
        ("v3_host_audit", "v3_host_audit_sha256"),
    ):
        validate_binding(cfg, key, digest, faults)
    v3_cfg = read_bound(cfg, "v3_config", "v3_config_sha256", faults)
    v3_report = read_bound(cfg, "v3_report", "v3_report_sha256", faults)
    v3_audit = read_bound(cfg, "v3_audit", "v3_audit_sha256", faults)
    host = read_bound(cfg, "v3_host_report", "v3_host_report_sha256", faults)
    host_audit = read_bound(cfg, "v3_host_audit", "v3_host_audit_sha256", faults)
    if v3_report.get("trigger_state") != "GREEN" or v3_audit.get("trigger_state") != "GREEN":
        faults.append("v3_static_boundary_invalid")
    if host.get("state") != "INCONCLUSIVE_EXPERIMENT_HOST_OPERABILITY" or host_audit.get("trigger_state") != "GREEN":
        faults.append("v3_host_disposition_invalid")
    if host.get("nine_task_screen_authorized") is not False:
        faults.append("v3_screen_must_remain_closed")
    store = read_bound(v3_cfg, "k2_store", "k2_store_sha256", faults)
    k2_report = read_bound(v3_cfg, "k2_report", "k2_report_sha256", faults)
    visible = {str(row.get("request_id")): row for row in p2a.dicts(k2_report.get("rows"))}
    maximum_chars = int(cfg.get("source_chunk_maximum_chars") or 0)
    context_window = int(v3_cfg.get("model_context_window_tokens") or 0)
    packet_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    total_files = total_chunks = reconstructed_files = 0
    for ordinal, store_row in enumerate(p2a.dicts(store.get("rows")), start=1):
        request_id = str(store_row.get("request_id") or "")
        request = str(p2a.mapping(p2a.mapping(visible.get(request_id)).get("candidate_surface")).get("natural_language_request") or "")
        selector = p2a.mapping(store_row.get("selector"))
        request_terms = p2a.strings(selector.get("request_terms"))
        files = v3.load_text_pages(store_row)
        chunks: list[dict[str, Any]] = []
        for file_path in sorted(files):
            file_chunks = chunk_text(file_path, files[file_path], maximum_chars)
            reconstructed_files += int("".join(row["text"] for row in file_chunks) == files[file_path])
            total_files += 1
            total_chunks += len(file_chunks)
            chunks.extend(file_chunks)
        for row in chunks:
            searchable = f"{row['path']}\n{row['text']}".lower()
            row["matched_request_terms"] = [term for term in request_terms if term in searchable]
            row["score"] = len(row["matched_request_terms"])
        vcm_chunks = v3.minimum_request_cover(chunks)
        ordinary_chunks = [row for row in chunks if int(row.get("score") or 0) > 0]
        rotation = list(v3.ROUTES[ordinal - 1:] + v3.ROUTES[:ordinal - 1])
        arms: list[dict[str, Any]] = []
        for order, route in enumerate(rotation, start=1):
            pages = v3.route_pages(route, vcm_chunks, ordinary_chunks, chunks, request_id)
            context = render_chunk_context(route, pages, request_id)
            prompt = v3.render_prompt(request, route, context)
            measurement = v3.normalize_token_measurement(token_counter(v3.SYSTEM_PROMPT, prompt))
            tokens = measurement.get("exact_tokens")
            lower = int(measurement.get("lower_bound_tokens") or 0)
            eligible = tokens is not None and int(tokens) < context_window
            information = v3.page_information_receipt(pages)
            arm = {
                "request_id": request_id, "row_ordinal": ordinal,
                "within_row_order": order, "route": route,
                "candidate_execution_mode": "integrated_vcm_packet" if route == "governed_vcm" else "direct_parent_only_control",
                "context_chunk_count": len(pages),
                "context_source_bytes": sum(int(row.get("bytes") or 0) for row in pages),
                "context_information_sha256": information,
                "context_chunk_receipts": [{k: row.get(k) for k in ("path", "chunk_id", "start_char", "end_char", "bytes", "sha256")} for row in pages],
                "prompt_sha256": p2a.sha256_text(prompt),
                "exact_chat_prompt_tokens": tokens,
                "prompt_token_lower_bound": lower,
                "prompt_token_measurement": measurement.get("kind"),
                "physical_context_residual_tokens": context_window - int(tokens) if tokens is not None else None,
                "physically_addressable": eligible,
                "ineligible_reason": "" if eligible else "prompt_reaches_or_exceeds_physical_context_boundary",
                "project_selected_quality_token_cap": None,
            }
            arms.append(arm); packet_rows.append(arm)
        by_route = {row["route"]: row for row in arms}
        flat = by_route["information_matched_flat_direct_context"]
        governed = by_route["governed_vcm"]
        matched_pair = (
            flat["context_information_sha256"] == governed["context_information_sha256"]
            and flat["physically_addressable"] and governed["physically_addressable"]
        )
        if flat["context_information_sha256"] != governed["context_information_sha256"]:
            faults.append(f"flat_vcm_information_mismatch:{request_id}")
        row_rows.append({
            "request_id": request_id, "row_ordinal": ordinal,
            "source_file_count": len(files), "source_chunk_count": len(chunks),
            "vcm_chunk_count": len(vcm_chunks), "ordinary_chunk_count": len(ordinary_chunks),
            "grounded_request_term_count": len(v3.grounded_terms(chunks)),
            "vcm_flat_physically_addressable_matched_pair": bool(matched_pair),
            "route_order": rotation, "arms": arms,
        })
    pair_count = sum(row["vcm_flat_physically_addressable_matched_pair"] for row in row_rows)
    if reconstructed_files != total_files:
        faults.append("source_reconstruction_incomplete")
    if len(row_rows) != 6 or len(packet_rows) != 36:
        faults.append("denominator_invalid")
    if pair_count != 6:
        faults.append("vcm_flat_physical_pair_incomplete")
    consumed_hashes = {str(call.get("prompt_sha256") or "") for call in p2a.dicts(host.get("calls"))}
    consumed_identity_count = 0
    for row in packet_rows:
        consumed_identity = row.get("prompt_sha256") in consumed_hashes
        row["consumed_v3_prompt_identity"] = consumed_identity
        row["new_host_call_authorized"] = False
        consumed_identity_count += int(consumed_identity)
    report = {
        "policy": POLICY, "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "VCM_V4_CHUNKED_ROUTE_PREFLIGHT_GREEN_HOST_LADDER_REQUIRED" if not faults else "VCM_V4_CHUNKED_ROUTE_PREFLIGHT_INVALID",
        "faults": sorted(set(faults)), "config": identity(path),
        "row_count": len(row_rows), "route_count": len(v3.ROUTES), "packet_count": len(packet_rows),
        "source_file_count": total_files, "source_chunk_count": total_chunks,
        "reconstructed_source_file_count": reconstructed_files,
        "vcm_flat_physically_addressable_matched_pair_count": pair_count,
        "consumed_v3_prompt_identity_count": consumed_identity_count,
        "rows": row_rows,
        "consumed_v3_prompt_reruns_authorized": False,
        "new_local_model_calls_authorized": 0,
        "local_model_calls": 0, "hidden_evaluator_calls": 0,
        "external_reference_calls": 0, "teacher_calls": 0,
        "project_selected_quality_token_cap": None,
        "maximum_inference": cfg.get("maximum_inference"),
    }
    packets = {
        "policy": "project_theseus_vcm_v4_chunked_packet_manifest_v2",
        "created_utc": p2a.now(), "rows": packet_rows,
        "raw_context_stored": False, "local_model_calls": 0,
        "external_reference_calls": 0,
    }
    return report, packets


def render_chunk_context(route: str, rows: list[dict[str, Any]], request_id: str) -> str:
    blocks = []
    for row in rows:
        text = str(row.get("text") or "")
        if route == "hierarchical_summary_or_prompt_compression_same_parent_store_and_context_opportunity":
            terms = p2a.strings(row.get("matched_request_terms"))
            text = "\n".join(line for line in text.splitlines() if line.strip() and (line.lstrip().startswith(("#", "//", "/*", "*", "use ", "import ", "from ", "def ", "class ", "fn ", "pub ")) or any(term in line.lower() for term in terms)))
        blocks.append(f"FILE_CHUNK {row['chunk_id']}\n<<<\n{text}\n>>>")
    body = "\n\n".join(blocks)
    return f"VCM_PACKET request={request_id} authority=licensed_parent_snapshot_read effect_root=repository\n{body}" if route == "governed_vcm" else body


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG)); args = parser.parse_args()
    path = p2a.resolve(args.config); cfg = p2a.read_json(path)
    report, packets = build(path, token_counter=v3.exact_token_counter(p2a.read_json(p2a.resolve(str(cfg.get("v3_config"))))))
    packets_path = p2a.resolve(str(cfg.get("packets_out") or "")); p2a.write_json(packets_path, packets)
    report["packets_artifact"] = identity(packets_path); p2a.write_json(p2a.resolve(str(cfg.get("report") or "")), report)
    print(json.dumps({k: report.get(k) for k in ("trigger_state", "state", "faults", "row_count", "packet_count", "source_chunk_count", "vcm_flat_physically_addressable_matched_pair_count", "local_model_calls")}, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


if __name__ == "__main__": raise SystemExit(main())
