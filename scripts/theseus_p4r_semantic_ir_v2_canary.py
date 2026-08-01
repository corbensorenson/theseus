#!/usr/bin/env python3
"""Learned non-claim canary for labeled Semantic-IR v2 transport mechanics."""

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
import theseus_semantic_ir_v2 as ir_v2  # noqa: E402


POLICY = "project_theseus_p4r_semantic_ir_v2_learned_mechanics_canary_v1"
CONFIG_POLICY = "project_theseus_p4r_semantic_ir_v2_mechanics_canary_config_v1"
MODEL_CONTEXT_TOKENS = 262144


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/theseus_p4r_semantic_ir_v2_mechanics_canary.json"
    )
    parser.add_argument(
        "--out", default="reports/theseus_p4r_semantic_ir_v2_mechanics_canary.json"
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = audit_config(config_path) if args.audit_only else run_canary(config_path)
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


def audit_config(path: Path) -> dict[str, Any]:
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") != CONFIG_POLICY:
        faults.append("config_policy_invalid")
    if value.get("state") != "FROZEN_BEFORE_LEARNED_MECHANICS_CALLS":
        faults.append("config_not_frozen")
    for name, expected_key in (
        ("parser", "parser_sha256"),
        ("runner", "runner_sha256"),
        ("base_local_instrument", "base_local_instrument_sha256"),
        ("terminal_p4r_disposition", "terminal_p4r_disposition_sha256"),
    ):
        owner = p2a.resolve(str(value.get(name) or ""))
        if p2a.sha256_file(owner) != str(value.get(expected_key) or ""):
            faults.append(f"{name}_digest_mismatch")
    disposition = p2a.read_json(
        p2a.resolve(str(value.get("terminal_p4r_disposition") or ""))
    )
    if disposition.get("scientific_status") != "INCONCLUSIVE_IMPLEMENTATION":
        faults.append("predecessor_disposition_invalid")
    base = p2a.read_json(p2a.resolve(str(value.get("base_local_instrument") or "")))
    frozen = p2a.mapping(base.get("frozen_model"))
    if frozen.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if int(frozen.get("model_declared_context_window_tokens") or 0) != MODEL_CONTEXT_TOKENS:
        faults.append("model_context_binding_invalid")
    cases = p2a.dicts(value.get("cases"))
    if len(cases) != 3 or len({row.get("case_id") for row in cases}) != 3:
        faults.append("case_denominator_invalid")
    for row in cases:
        if not all(str(row.get(key) or "") for key in (
            "case_id", "natural_request", "source", "target_node_type", "expected_source"
        )):
            faults.append(f"case_incomplete:{row.get('case_id')}")
    return {
        "policy": "project_theseus_p4r_semantic_ir_v2_canary_audit_v1",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "config_sha256": p2a.sha256_file(path),
        "counters": p2a.zero_counters(),
    }


def run_canary(
    config_path: Path,
    *, session_factory: Callable[..., Any] = local_backend.PersistentLocalInferenceSession,
) -> dict[str, Any]:
    audit = audit_config(config_path)
    if audit.get("trigger_state") != "GREEN":
        return {
            "policy": POLICY,
            "created_utc": p2a.now(),
            "trigger_state": "RED",
            "state": "CONFIG_INVALID",
            "faults": ["config_audit_red"],
            "config_audit": audit,
            "counters": p2a.zero_counters(),
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
        session_id="p4r-semantic-ir-v2-mechanics",
        completion_predicate=ir_v2.complete,
    )
    if not session.ready:
        return {
            "policy": POLICY,
            "created_utc": p2a.now(),
            "trigger_state": "RED",
            "state": "BACKEND_INVALID",
            "faults": ["persistent_backend_not_ready", *list(session.faults)],
            "config_audit": audit,
            "counters": p2a.zero_counters(),
        }
    rows: list[dict[str, Any]] = []
    with assistant_runtime.bind_local_inference_runner(session.runtime_runner):
        for case in p2a.dicts(config.get("cases")):
            rows.append(run_case(case, base))
    parse_and_lower = sum(int(row["parse_and_lower"]) for row in rows)
    verified = sum(int(row["verified"]) for row in rows)
    ceiling_hits = sum(int(row["safety_ceiling_hit"]) for row in rows)
    route_ready = all(row["route_integrity_ready"] for row in rows)
    termination_ready = all(
        row["termination_reason"] in {"parser_complete", "model_eos"}
        for row in rows
    )
    passed = (
        parse_and_lower == verified == 3
        and session.model_load_count == 1
        and session.inference_calls == 3
        and ceiling_hits == 0
        and route_ready
        and termination_ready
    )
    faults = [] if passed else ["learned_transport_mechanics_floor_not_met"]
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if passed else "YELLOW",
        "state": (
            "LEARNED_V2_TRANSPORT_MECHANICS_GREEN"
            if passed else "INCONCLUSIVE_IMPLEMENTATION"
        ),
        "faults": faults,
        "config_sha256": p2a.sha256_file(config_path),
        "config_audit": audit,
        "parse_and_lower": f"{parse_and_lower}/3",
        "verified": f"{verified}/3",
        "model_calls": session.inference_calls,
        "model_loads": session.model_load_count,
        "safety_ceiling_hits": ceiling_hits,
        "cases": rows,
        "scope": "Non-claim hand-authored transport mechanics only. Passing cannot support cognitive compilation or qualify a fresh P4 denominator by itself.",
        "next_gate": "Intervention and dependency-local repair canary, then a fresh source-disjoint P4 denominator only if all mechanics gates pass.",
        "counters": p2a.zero_counters(),
    }


def run_case(case: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    case_id = p2a.safe_slug(str(case.get("case_id") or "case"))
    with tempfile.TemporaryDirectory(prefix=f"theseus-p4r-v2-{case_id}-") as tmp:
        root = Path(tmp)
        source_path = root / "sample.py"
        source_path.write_text(str(case.get("source") or ""), encoding="utf-8")
        task = build_task(case)
        symbols = p4.semantic_symbol_table(root, task)
        target = select_target(symbols, str(case.get("target_node_type") or ""))
        prompt = render_prompt(case, task, symbols, target)
        call = p2a.runtime_call(
            route_integrity.DIRECT_MODE,
            f"p4r_v2_mechanics_{case_id}",
            1,
            prompt,
            MODEL_CONTEXT_TOKENS,
            str(base.get("runtime_config") or ""),
        )
        parsed = ir_v2.parse(str(call.get("assistant_text") or ""), task, root)
        apply_faults = p2a.apply_actions(root, p2a.dicts(parsed.get("actions")))
        observed = source_path.read_text(encoding="utf-8")
        verified = not parsed.get("faults") and not apply_faults and observed == str(
            case.get("expected_source") or ""
        )
        runtime = p2a.mapping(call.get("runtime_report"))
        backend_path = p2a.resolve(
            str(p2a.mapping(runtime.get("generation_backend")).get("out") or "")
        )
        backend = p2a.read_json(backend_path)
        metrics = p2a.mapping(backend.get("metrics"))
        return {
            "case_id": case.get("case_id"),
            "target_node_id": target.get("id"),
            "target_node_sha256": target.get("sha256"),
            "parse_and_lower": not parsed.get("faults") and bool(parsed.get("actions")),
            "parse_faults": p2a.strings(parsed.get("faults")),
            "apply_faults": apply_faults,
            "verified": verified,
            "observed_source_sha256": p2a.sha256_text(observed),
            "expected_source_sha256": p2a.sha256_text(str(case.get("expected_source") or "")),
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


def build_task(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_effect_paths": ["sample.py"],
        "candidate_visible_context": {
            "reads": [{"path": "sample.py", "start_line": 1, "end_line": 1000}]
        },
        "semantic_ir_contract": {"maximum_symbol_nodes": 80},
        "obligations": p2a.dicts(case.get("obligations")),
        "obligation_dependencies": p2a.dicts(case.get("obligation_dependencies")),
    }


def select_target(symbols: dict[str, Any], node_type: str) -> dict[str, Any]:
    rows = [row for row in p2a.dicts(symbols.get("nodes")) if row.get("node_type") == node_type]
    if len(rows) != 1:
        raise p4.P4Fault(f"mechanics_target_not_unique:{node_type}:{len(rows)}")
    return rows[0]


def render_prompt(
    case: dict[str, Any], task: dict[str, Any], symbols: dict[str, Any],
    target: dict[str, Any],
) -> str:
    obligations = "\n".join(
        f"{row['id']} {str(row['kind']).upper()}: {row['text']}"
        for row in p2a.dicts(task.get("obligations"))
    )
    return (
        "This is a typed transport mechanics task. Return only one complete labeled Semantic IR artifact.\n"
        f"Request: {case.get('natural_request')}\n\n"
        + ir_v2.grammar()
        + "\n\nCopy SOURCE, PATH, NODE, and NODE_SHA exactly. Use exactly one UNIT and reference all obligations in order. LOSS must be NONE. Replacement source is the selected AST node text without surrounding leading indentation; the lowerer preserves the node's original prefix and suffix.\n"
        f"\nSOURCE {symbols['source_digest']}\n"
        f"TARGET PATH {target['path']}\nTARGET NODE {target['id']}\n"
        f"TARGET NODE_SHA {target['sha256']}\n"
        f"TARGET TYPE {target['node_type']}\n"
        f"OBLIGATIONS\n{obligations}\n"
        f"SOURCE_TEXT\n{case.get('source')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
