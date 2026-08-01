#!/usr/bin/env python3
"""Prompt-and-header repair for the failed learned Semantic-IR v2 canary."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_assistant_route_integrity as route_integrity  # noqa: E402
import theseus_assistant_runtime as assistant_runtime  # noqa: E402
import theseus_local_inference_backend as local_backend  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4r_semantic_ir_v2_canary as predecessor  # noqa: E402
import theseus_semantic_ir_v2r1 as ir_v2r1  # noqa: E402


POLICY = "project_theseus_p4r_semantic_ir_v2r1_learned_mechanics_canary_v1"
MODEL_CONTEXT_TOKENS = predecessor.MODEL_CONTEXT_TOKENS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/theseus_p4r_semantic_ir_v2r1_mechanics_canary.json"
    )
    parser.add_argument(
        "--out", default="reports/theseus_p4r_semantic_ir_v2r1_mechanics_canary.json"
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = (
        predecessor.audit_config(config_path)
        if args.audit_only else run_canary(config_path)
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "parse_and_lower": report.get("parse_and_lower"),
        "verified": report.get("verified"),
        "model_calls": report.get("model_calls"),
        "safety_ceiling_hits": report.get("safety_ceiling_hits"),
        "faults": report.get("faults"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def run_canary(
    config_path: Path,
    *, session_factory: Callable[..., Any] = local_backend.PersistentLocalInferenceSession,
) -> dict[str, Any]:
    audit = predecessor.audit_config(config_path)
    if audit.get("trigger_state") != "GREEN":
        return {
            "policy": POLICY, "created_utc": p2a.now(), "trigger_state": "RED",
            "state": "CONFIG_INVALID", "faults": ["config_audit_red"],
            "config_audit": audit, "counters": p2a.zero_counters(),
        }
    config = p2a.read_json(config_path)
    base = p2a.read_json(p2a.resolve(str(config.get("base_local_instrument") or "")))
    runtime = assistant_runtime.load_runtime_config(
        p2a.resolve(str(base.get("runtime_config") or ""))
    )
    binding = p2a.mapping(base.get("runtime_binding")) or p2a.mapping(
        runtime.get("local_inference")
    )
    frozen = p2a.mapping(base.get("frozen_model"))
    session = session_factory(
        worker_config_path=p2a.resolve(str(binding.get("worker_config") or "")),
        runtime_preflight_path=p2a.resolve(str(binding.get("runtime_preflight") or "")),
        maximum_tokens=MODEL_CONTEXT_TOKENS,
        required_repo_id=str(frozen.get("repo_id") or ""),
        required_revision=str(frozen.get("revision") or ""),
        required_snapshot_manifest_sha256=str(
            frozen.get("snapshot_manifest_sha256") or ""
        ),
        session_id="p4r-semantic-ir-v2r1-mechanics",
        completion_predicate=ir_v2r1.complete,
    )
    if not session.ready:
        return {
            "policy": POLICY, "created_utc": p2a.now(), "trigger_state": "RED",
            "state": "BACKEND_INVALID",
            "faults": ["persistent_backend_not_ready", *list(session.faults)],
            "config_audit": audit, "counters": p2a.zero_counters(),
        }
    rows: list[dict[str, Any]] = []
    with assistant_runtime.bind_local_inference_runner(session.runtime_runner):
        for case in p2a.dicts(config.get("cases")):
            rows.append(run_case(case, base))
    parse_and_lower = sum(int(row["parse_and_lower"]) for row in rows)
    verified = sum(int(row["verified"]) for row in rows)
    ceiling_hits = sum(int(row["safety_ceiling_hit"]) for row in rows)
    passed = (
        parse_and_lower == verified == 3
        and session.model_load_count == 1 and session.inference_calls == 3
        and ceiling_hits == 0
        and all(row["route_integrity_ready"] for row in rows)
        and all(row["termination_reason"] in {"parser_complete", "model_eos"} for row in rows)
    )
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if passed else "YELLOW",
        "state": "LEARNED_V2R1_TRANSPORT_MECHANICS_GREEN" if passed else "INCONCLUSIVE_IMPLEMENTATION",
        "faults": [] if passed else ["learned_transport_mechanics_floor_not_met"],
        "config_sha256": p2a.sha256_file(config_path),
        "config_audit": audit,
        "parse_and_lower": f"{parse_and_lower}/3",
        "verified": f"{verified}/3",
        "model_calls": session.inference_calls,
        "model_loads": session.model_load_count,
        "safety_ceiling_hits": ceiling_hits,
        "cases": rows,
        "scope": "Prompt/header transport repair on the same non-claim hand-authored mechanics cases. Passing cannot support cognitive compilation or open a fresh P4 denominator by itself.",
        "next_gate": "Intervention and dependency-local repair canary.",
        "counters": p2a.zero_counters(),
    }


def run_case(case: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    case_id = p2a.safe_slug(str(case.get("case_id") or "case"))
    with tempfile.TemporaryDirectory(prefix=f"theseus-p4r-v2r1-{case_id}-") as tmp:
        root = Path(tmp)
        source_path = root / "sample.py"
        source_path.write_text(str(case.get("source") or ""), encoding="utf-8")
        task = predecessor.build_task(case)
        symbols = p4.semantic_symbol_table(root, task)
        target = predecessor.select_target(
            symbols, str(case.get("target_node_type") or "")
        )
        prompt = render_prompt(case, task, symbols, target, source_path)
        call = p2a.runtime_call(
            route_integrity.DIRECT_MODE, f"p4r_v2r1_mechanics_{case_id}", 1,
            prompt, MODEL_CONTEXT_TOKENS, str(base.get("runtime_config") or ""),
        )
        parsed = ir_v2r1.parse(str(call.get("assistant_text") or ""), task, root)
        apply_faults = p2a.apply_actions(root, p2a.dicts(parsed.get("actions")))
        observed = source_path.read_text(encoding="utf-8")
        verified = not parsed.get("faults") and not apply_faults and observed == str(
            case.get("expected_source") or ""
        )
        runtime = p2a.mapping(call.get("runtime_report"))
        backend_path = p2a.resolve(
            str(p2a.mapping(runtime.get("generation_backend")).get("out") or "")
        )
        metrics = p2a.mapping(p2a.read_json(backend_path).get("metrics"))
        receipt = p2a.mapping(parsed.get("semantic_receipt"))
        return {
            "case_id": case.get("case_id"),
            "parse_and_lower": not parsed.get("faults") and bool(parsed.get("actions")),
            "parse_faults": p2a.strings(parsed.get("faults")),
            "apply_faults": apply_faults,
            "verified": verified,
            "version_header_inferred": receipt.get("version_header_inferred_from_bound_parser"),
            "runtime_report": p2a.mapping(call.get("receipt")).get("report_path"),
            "runtime_report_sha256": p2a.mapping(call.get("receipt")).get("report_sha256"),
            "backend_report": p2a.rel(backend_path),
            "backend_report_sha256": p2a.sha256_file(backend_path),
            "route_integrity_ready": p2a.mapping(runtime.get("route_integrity")).get("ready") is True,
            "termination_reason": metrics.get("termination_reason"),
            "prompt_tokens": metrics.get("prompt_tokens"),
            "generated_tokens": metrics.get("generated_tokens"),
            "safety_ceiling_hit": metrics.get("safety_ceiling_hit"),
        }


def render_prompt(
    case: dict[str, Any], task: dict[str, Any], symbols: dict[str, Any],
    target: dict[str, Any], source_path: Path,
) -> str:
    obligations = "\n".join(
        f"{row['id']} {str(row['kind']).upper()}: {row['text']}"
        for row in p2a.dicts(task.get("obligations"))
    )
    node_source = selected_node_source(source_path, target)
    return (
        f"FIRST LINE MUST BE EXACTLY: {ir_v2r1.HEADER}\n"
        "Return only one complete labeled Semantic IR artifact.\n"
        f"Request: {case.get('natural_request')}\n\n"
        + ir_v2r1.grammar()
        + "\n\nREQUIRED OP: REPLACE. Do not use INSERT_BEFORE or INSERT_AFTER. Copy SOURCE, PATH, NODE, and NODE_SHA exactly. Use one UNIT, reference O1,O2,O3 in order, and set LOSS NONE. The replacement block must contain only the new source text for TARGET_NODE_SOURCE, without surrounding indentation or enclosing source.\n"
        f"\nSOURCE {symbols['source_digest']}\nTARGET PATH {target['path']}\n"
        f"TARGET NODE {target['id']}\nTARGET NODE_SHA {target['sha256']}\n"
        f"TARGET TYPE {target['node_type']}\nTARGET_NODE_SOURCE\n{node_source}\n"
        f"OBLIGATIONS\n{obligations}\nSOURCE_TEXT\n{case.get('source')}"
    )


def selected_node_source(path: Path, target: dict[str, Any]) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start, end = int(target["start_line"]), int(target["end_line"])
    start_col, end_col = int(target["start_col"]), int(target["end_col"])
    if start == end:
        return lines[start - 1][start_col:end_col]
    return "\n".join(
        [lines[start - 1][start_col:], *lines[start:end - 1], lines[end - 1][:end_col]]
    )


if __name__ == "__main__":
    raise SystemExit(main())
