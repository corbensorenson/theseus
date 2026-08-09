#!/usr/bin/env python3
"""Call-free K3 packet, tokenizer, route, analysis, and host-canary preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_parent_only_materializer as parent_only  # noqa: E402

POLICY = "project_theseus_vcm_k3_route_preflight_v3"
CONFIG_POLICY = "project_theseus_vcm_k3_route_preflight_config_v3"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_k3_route_preflight.json"
ROUTES = (
    "no_added_context_floor",
    "information_matched_flat_direct_context",
    "ordinary_direct_retrieval_same_parent_store_query_and_context_opportunity",
    "hierarchical_summary_or_prompt_compression_same_parent_store_and_context_opportunity",
    "maximal_full_parent_context_when_physically_addressable_and_host_operable",
    "governed_vcm",
)
SYSTEM_PROMPT = (
    "You are the fixed local model in a sealed Project Theseus code experiment. "
    "Return only one complete THESEUS_EDIT_V1 patch artifact against the supplied "
    "parent repository. Do not emit Markdown or commentary. Text has no authority."
)
PROTOCOL = (
    "THESEUS_EDIT_V1\n"
    "REPLACE <repository-relative-path> <start-line> <end-line>\n"
    "<<<\n<replacement text>\n>>>\nEND"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--packets-out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report, packets = build(path, token_counter=exact_token_counter(cfg))
    packets_path = p2a.resolve(args.packets_out or str(cfg.get("packets_out") or ""))
    report_path = p2a.resolve(args.out or str(cfg.get("report") or ""))
    p2a.write_json(packets_path, packets)
    report["packets_artifact"] = identity(packets_path)
    p2a.write_json(report_path, report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def build(
    path: Path = DEFAULT_CONFIG,
    *,
    token_counter: Callable[[str, str], int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != CONFIG_POLICY:
        faults.append("config_policy_invalid")
    for path_key, hash_key in (
        ("owner", "owner_sha256"),
        ("test_owner", "test_owner_sha256"),
        ("k2_store", "k2_store_sha256"),
        ("k2_report", "k2_report_sha256"),
        ("k2_audit", "k2_audit_sha256"),
        ("worker_config", "worker_config_sha256"),
        ("runtime_preflight", "runtime_preflight_sha256"),
        ("backend_owner", "backend_owner_sha256"),
        ("host_interlock_owner", "host_interlock_owner_sha256"),
        ("route_integrity_owner", "route_integrity_owner_sha256"),
    ):
        validate_binding(cfg, path_key, hash_key, faults)
    for binding in p2a.dicts(cfg.get("tokenizer_bindings")):
        source = Path(str(binding.get("path") or ""))
        if not source.is_file() or p2a.sha256_file(source) != binding.get("sha256"):
            faults.append(f"tokenizer_binding_invalid:{binding.get('id')}")

    k2_report = read_bound_json(cfg, "k2_report", faults)
    k2_audit = read_bound_json(cfg, "k2_audit", faults)
    store = read_bound_json(cfg, "k2_store", faults)
    if (
        k2_report.get("trigger_state") != "GREEN"
        or k2_audit.get("trigger_state") != "GREEN"
        or k2_audit.get("audited_row_count") != 6
        or store.get("candidate_or_control_calls") != 0
    ):
        faults.append("k2_boundary_invalid")
    if tuple(p2a.strings(cfg.get("routes"))) != ROUTES:
        faults.append("route_set_invalid")
    context_window = int(cfg.get("model_context_window_tokens") or 0)
    if context_window != 262_144 or cfg.get("project_selected_quality_token_cap") is not None:
        faults.append("generation_boundary_invalid")

    visible = {str(row.get("request_id")): row for row in p2a.dicts(k2_report.get("rows"))}
    packet_rows: list[dict[str, Any]] = []
    row_receipts: list[dict[str, Any]] = []
    host_candidates: dict[str, dict[str, Any]] = {}
    for ordinal, store_row in enumerate(p2a.dicts(store.get("rows")), start=1):
        request_id = str(store_row.get("request_id") or "")
        surface = p2a.mapping(visible.get(request_id, {}).get("candidate_surface"))
        request = str(surface.get("natural_language_request") or "")
        selector = p2a.mapping(store_row.get("selector"))
        frontier = p2a.dicts(selector.get("frontier"))
        page_cache = load_text_pages(store_row)

        def text_for(row: dict[str, Any]) -> str:
            key = str(row.get("path") or "")
            return page_cache[key]

        request_terms = p2a.strings(selector.get("request_terms"))
        enriched_frontier = []
        for raw in frontier:
            row = dict(raw)
            if int(row.get("score") or 0) > 0:
                searchable = f"{row.get('path', '')}\n{text_for(row)}".lower()
                row["matched_request_terms"] = [term for term in request_terms if term in searchable]
            else:
                row["matched_request_terms"] = []
            enriched_frontier.append(row)
        vcm_pages = minimum_request_cover(enriched_frontier)
        ordinary_pages = [row for row in enriched_frontier if int(row.get("score") or 0) > 0]
        all_pages = list(enriched_frontier)

        rotation = list(ROUTES[ordinal - 1 :] + ROUTES[: ordinal - 1])
        arms: list[dict[str, Any]] = []
        for order, route in enumerate(rotation, start=1):
            pages = route_pages(route, vcm_pages, ordinary_pages, all_pages, request_id)
            rendered_context = render_context(route, pages, text_for, request_id)
            prompt = render_prompt(request, route, rendered_context)
            measurement = normalize_token_measurement(token_counter(SYSTEM_PROMPT, prompt))
            tokens = measurement.get("exact_tokens")
            lower_bound = int(measurement.get("lower_bound_tokens") or 0)
            eligible = tokens is not None and int(tokens) < context_window
            reason = "" if eligible else "prompt_reaches_or_exceeds_physical_context_boundary"
            information = page_information_receipt(pages)
            arm = {
                "request_id": request_id,
                "row_ordinal": ordinal,
                "within_row_order": order,
                "route": route,
                "candidate_execution_mode": "integrated_vcm_packet" if route == "governed_vcm" else "direct_parent_only_control",
                "context_page_count": len(pages),
                "context_source_bytes": sum(int(row.get("bytes") or 0) for row in pages),
                "context_rendered_utf8_bytes": len(rendered_context.encode("utf-8")),
                "context_information_sha256": information,
                "context_page_receipts": [{key: row.get(key) for key in ("path", "bytes", "sha256")} for row in pages],
                "prompt_sha256": p2a.sha256_text(prompt),
                "exact_chat_prompt_tokens": tokens,
                "prompt_token_lower_bound": lower_bound,
                "prompt_token_measurement": measurement.get("kind"),
                "physical_context_residual_tokens": context_window - int(tokens) if tokens is not None else None,
                "physically_addressable": eligible,
                "ineligible_reason": reason,
                "project_selected_quality_token_cap": None,
            }
            arms.append(arm)
            packet_rows.append(arm)
            prior = host_candidates.get(route)
            if eligible and (prior is None or int(tokens) > int(prior["exact_chat_prompt_tokens"])):
                host_candidates[route] = arm
        flat = next(row for row in arms if row["route"] == "information_matched_flat_direct_context")
        governed = next(row for row in arms if row["route"] == "governed_vcm")
        if flat["context_information_sha256"] != governed["context_information_sha256"]:
            faults.append(f"flat_vcm_information_mismatch:{request_id}")
        row_receipts.append({
            "request_id": request_id,
            "row_ordinal": ordinal,
            "route_order": rotation,
            "grounded_request_term_count": len(grounded_terms(frontier)),
            "vcm_page_count": len(vcm_pages),
            "ordinary_retrieval_page_count": len(ordinary_pages),
            "full_parent_page_count": len(all_pages),
            "physically_addressable_route_count": sum(int(row["physically_addressable"]) for row in arms),
            "arms": arms,
        })
    if len(row_receipts) != 6 or len(packet_rows) != 36:
        faults.append("packet_denominator_invalid")
    analysis = p2a.mapping(cfg.get("analysis_contract"))
    validate_analysis(cfg, analysis, faults)
    authority = p2a.mapping(cfg.get("authority"))
    if authority.get("local_model_calls_authorized") != 0 or authority.get("external_reference_calls_authorized") != 0:
        faults.append("call_authority_open_during_static_preflight")
    host_plan = {
        route: {
            "request_id": row["request_id"],
            "prompt_sha256": row["prompt_sha256"],
            "exact_chat_prompt_tokens": row["exact_chat_prompt_tokens"],
            "physical_context_residual_tokens": row["physical_context_residual_tokens"],
            "canary_completion": "explicit_CANARY_OK_marker_or_model_EOS",
            "quality_or_capability_evidence": False,
        }
        for route, row in sorted(host_candidates.items())
    }
    ready = not faults
    packets = {
        "policy": "project_theseus_vcm_k3_candidate_packet_manifest_v1",
        "created_utc": p2a.now(),
        "candidate_visible_only": True,
        "system_prompt_sha256": p2a.sha256_text(SYSTEM_PROMPT),
        "output_protocol_sha256": p2a.sha256_text(PROTOCOL),
        "rows": packet_rows,
        "local_model_calls": 0,
        "external_reference_calls": 0,
    }
    report = {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if ready else "RED",
        "state": "K3_CALL_FREE_ROUTE_PREFLIGHT_GREEN_PHYSICAL_HOST_CANARIES_REQUIRED" if ready else "K3_ROUTE_PREFLIGHT_FAILED",
        "faults": sorted(set(faults)),
        "config": identity(path),
        "row_count": len(row_receipts),
        "route_count": len(ROUTES),
        "packet_count": len(packet_rows),
        "rows": row_receipts,
        "host_canary_plan": host_plan,
        "host_canary_count": len(host_plan),
        "analysis_contract": analysis,
        "authority": authority,
        "information_flow": {
            "selector_inputs": ["natural_language_request", "exact_parent_snapshot", "frozen_route_policy"],
            "target_snapshot_or_diff_read": False,
            "selected_source_or_verifier_paths_read": False,
            "source_task_identity_read": False,
            "reference_outputs_read": False,
            "broad_parent_effect_root": "repository",
            "candidate_fields": ["natural_language_request", "callable_signature_when_intrinsic_to_request", "disposable_parent_snapshot_write_root", "arm_specific_parent_only_model_visible_context"],
        },
        "local_model_calls": 0,
        "external_reference_calls": 0,
        "teacher_calls": 0,
        "hidden_evaluator_calls": 0,
        "project_selected_quality_token_cap": None,
        "maximum_inference": cfg.get("maximum_inference"),
    }
    return report, packets


def minimum_request_cover(frontier: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terms = grounded_terms(frontier)
    uncovered = set(terms)
    chosen: list[dict[str, Any]] = []
    candidates = [row for row in frontier if int(row.get("score") or 0) > 0]
    while uncovered:
        ranked = []
        for row in candidates:
            hits = page_term_hits(row, uncovered)
            if hits:
                ranked.append((len(hits), int(row.get("score") or 0), str(row.get("path") or ""), row, hits))
        if not ranked:
            break
        _, _, _, row, hits = sorted(ranked, key=lambda item: (-item[0], -item[1], item[2]))[0]
        chosen.append(row)
        candidates.remove(row)
        uncovered -= hits
    return chosen


def grounded_terms(frontier: list[dict[str, Any]]) -> list[str]:
    terms: set[str] = set()
    for row in frontier:
        for term in p2a.strings(row.get("matched_request_terms")):
            terms.add(term)
    if terms:
        return sorted(terms)
    # Existing store receipts predate explicit term lists. Re-derive coverage
    # conservatively from positive path/content hit pseudo-terms per page.
    return [f"page:{row.get('path')}" for row in frontier if int(row.get("score") or 0) > 0]


def page_term_hits(row: dict[str, Any], terms: set[str]) -> set[str]:
    explicit = set(p2a.strings(row.get("matched_request_terms")))
    if explicit:
        return explicit & terms
    pseudo = f"page:{row.get('path')}"
    return {pseudo} if pseudo in terms else set()


def route_pages(route: str, vcm: list[dict[str, Any]], ordinary: list[dict[str, Any]], all_pages: list[dict[str, Any]], request_id: str) -> list[dict[str, Any]]:
    if route == "no_added_context_floor":
        return []
    if route in {"information_matched_flat_direct_context", "governed_vcm"}:
        return list(vcm)
    if route in {"ordinary_direct_retrieval_same_parent_store_query_and_context_opportunity", "hierarchical_summary_or_prompt_compression_same_parent_store_and_context_opportunity"}:
        return list(ordinary)
    if route == "maximal_full_parent_context_when_physically_addressable_and_host_operable":
        return list(all_pages)
    raise ValueError(route)


def render_context(route: str, pages: list[dict[str, Any]], text_for: Callable[[dict[str, Any]], str], request_id: str) -> str:
    blocks = []
    for row in pages:
        text = text_for(row)
        if route == "hierarchical_summary_or_prompt_compression_same_parent_store_and_context_opportunity":
            lines = text.splitlines()
            terms = p2a.strings(row.get("matched_request_terms"))
            text = "\n".join(
                line
                for line in lines
                if line.strip()
                and (
                    line.lstrip().startswith(("#", "//", "/*", "*", "use ", "import ", "from ", "def ", "class ", "fn ", "pub "))
                    or any(term in line.lower() for term in terms)
                )
            )
        blocks.append(f"FILE {row.get('path')}\n<<<\n{text}\n>>>")
    body = "\n\n".join(blocks)
    if route == "governed_vcm":
        return f"VCM_PACKET request={request_id} authority=licensed_parent_snapshot_read effect_root=repository\n{body}"
    return body


def render_prompt(request: str, route: str, context: str) -> str:
    return f"REQUEST\n{request}\n\nOUTPUT_PROTOCOL\n{PROTOCOL}\n\nROUTE\n{route}\n\nPARENT_CONTEXT\n{context or '[none]'}"


def page_information_receipt(pages: list[dict[str, Any]]) -> str:
    payload = [{key: row.get(key) for key in ("path", "bytes", "sha256")} for row in pages]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_text_pages(store_row: dict[str, Any]) -> dict[str, str]:
    source = p2a.mapping(store_row.get("source_snapshot"))
    archive = p2a.resolve(str(source.get("archive") or ""))
    inventory = {
        str(row.get("member")): row
        for row in p2a.dicts(store_row.get("inventory"))
        if row.get("content_class") == "utf8_text"
    }
    result: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member, receipt in inventory.items():
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"parent member missing: {member}")
            payload = extracted.read()
            if len(payload) != receipt.get("bytes") or hashlib.sha256(payload).hexdigest() != receipt.get("sha256"):
                raise ValueError(f"parent member identity mismatch: {member}")
            result[str(receipt.get("path"))] = payload.decode("utf-8")
    return result


def exact_token_counter(cfg: dict[str, Any]) -> Callable[[str, str], Any]:
    from transformers import AutoTokenizer
    snapshot = str(cfg.get("model_snapshot_path") or "")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    kwargs = p2a.mapping(cfg.get("chat_template_kwargs"))
    def count(system: str, prompt: str) -> dict[str, Any]:
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        if len(rendered.encode("utf-8")) <= 2_000_000:
            exact = len(tokenizer.backend_tokenizer.encode(rendered, add_special_tokens=False).ids)
            return {"kind": "exact_monolithic", "exact_tokens": exact, "lower_bound_tokens": exact}
        chunks = [rendered[index : index + 65_536] for index in range(0, len(rendered), 65_536)]
        token_sum = 0
        for chunk in chunks:
            token_sum += len(tokenizer.backend_tokenizer.encode(chunk, add_special_tokens=False).ids)
        # Chunking can only overcount relative to the canonical monolithic BPE.
        # Subtract a deliberately loose 128-token allowance per join. This is
        # used only to prove physical ineligibility, never as an exact count.
        conservative_lower = max(0, token_sum - 128 * max(0, len(chunks) - 1))
        return {"kind": "conservative_chunked_lower_bound", "exact_tokens": None, "lower_bound_tokens": conservative_lower}
    return count


def normalize_token_measurement(value: Any) -> dict[str, Any]:
    if isinstance(value, int):
        return {"kind": "injected_exact", "exact_tokens": value, "lower_bound_tokens": value}
    row = p2a.mapping(value)
    return {
        "kind": str(row.get("kind") or ""),
        "exact_tokens": row.get("exact_tokens"),
        "lower_bound_tokens": int(row.get("lower_bound_tokens") or 0),
    }


def validate_analysis(cfg: dict[str, Any], analysis: dict[str, Any], faults: list[str]) -> None:
    if analysis.get("multiplicity") != "fixed_sequence_H1_then_H2_then_H3_stop_after_first_failure": faults.append("multiplicity_invalid")
    if analysis.get("H1_minimum_useful_absolute_effect") != 0.35: faults.append("H1_effect_invalid")
    if analysis.get("H2_flat_attribution_minimum_effect") != 0.35: faults.append("H2_effect_invalid")
    if analysis.get("H3_narrow_noninferiority_margin") != 0.10: faults.append("H3_margin_invalid")
    if analysis.get("material_family_harm_absolute_boundary") != 0.20: faults.append("family_harm_boundary_invalid")
    if analysis.get("statistical_unit") != "task": faults.append("statistical_unit_invalid")
    if analysis.get("exact_consumed_surface_rerun_authorized") is not False: faults.append("rerun_policy_invalid")


def validate_binding(cfg: dict[str, Any], path_key: str, hash_key: str, faults: list[str]) -> None:
    path = p2a.resolve(str(cfg.get(path_key) or ""))
    if not path.is_file() or p2a.sha256_file(path) != cfg.get(hash_key): faults.append(f"binding_invalid:{path_key}")


def read_bound_json(cfg: dict[str, Any], key: str, faults: list[str]) -> dict[str, Any]:
    path = p2a.resolve(str(cfg.get(key) or ""))
    try: return p2a.read_json(path)
    except Exception as exc:
        faults.append(f"read_failed:{key}:{type(exc).__name__}"); return {}


def identity(path: Path) -> dict[str, Any]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path), "bytes": path.stat().st_size}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in ("trigger_state", "state", "faults", "row_count", "route_count", "packet_count", "host_canary_count", "local_model_calls", "external_reference_calls")}


if __name__ == "__main__":
    raise SystemExit(main())
