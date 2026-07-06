"""Create governed multi-stream code-repair benchmark traces.

This is a concept port of the Multi-Stream LLM paper into Theseus runtime
pressure. It does not import the paper code, whose repo does not currently
advertise a license. Instead it builds local stream-table traces from existing
local synthetic code cases and keeps them quarantined as private pressure.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/multi_stream_policy.json")
    parser.add_argument("--cards-dir", default="benchmarks/cards")
    parser.add_argument("--cases-dir", default="data/multi_stream_benchmarks")
    parser.add_argument("--out", default="reports/multi_stream_trace_factory.json")
    parser.add_argument("--markdown-out", default="reports/multi_stream_trace_factory.md")
    parser.add_argument("--write-cards", action="store_true")
    args = parser.parse_args()

    policy = read_json(resolve(args.policy), {})
    streams = [str(item) for item in policy.get("streams", []) if str(item)]
    source_cases = load_source_cases(policy)
    max_cases = int(policy.get("max_cases") or 18)
    selected = source_cases[:max_cases]
    cases = [build_multistream_case(case, streams, index) for index, case in enumerate(selected)]
    card = build_card(policy, cases)

    cases_dir = resolve(args.cases_dir)
    card_manifest = resolve(str(card.get("case_manifest")))
    if not card_manifest.is_absolute():
        card_manifest = ROOT / card_manifest
    if not str(card_manifest).replace("\\", "/").startswith(str(cases_dir).replace("\\", "/")):
        card_manifest = cases_dir / f"{card['id']}.jsonl"
        card["case_manifest"] = rel(card_manifest)
        card["resource_pantry_path"] = rel(card_manifest)
        card["staged_path"] = rel(card_manifest)
    write_jsonl(card_manifest, cases)
    if args.write_cards:
        write_json(resolve(args.cards_dir) / f"{card['id']}.json", card)

    gates = [
        gate("source_cases_loaded", bool(source_cases), f"sources={len(source_cases)}"),
        gate("cases_generated", len(cases) >= int(policy.get("min_cases") or 1), f"cases={len(cases)}"),
        gate("stream_schema_present", all(case.get("streams") == streams for case in cases), ",".join(streams)),
        gate("case_manifest_written", card_manifest.exists(), rel(card_manifest)),
        gate("card_written", bool(args.write_cards and (resolve(args.cards_dir) / f"{card['id']}.json").exists()), card["id"]),
        gate("external_inference_zero", True, "deterministic local trace construction"),
        gate("public_comparator_quarantined", card.get("public_comparator_use") == "forbidden", str(card.get("public_comparator_use"))),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "RED"
    report = {
        "policy": "project_theseus_multi_stream_trace_factory_v1",
        "created_utc": now(),
        "config": rel(resolve(args.policy)),
        "trigger_state": trigger_state,
        "summary": {
            "cards": 1,
            "ready_cards": 1 if card.get("status") == "adapter_smoke_passed" else 0,
            "case_count": len(cases),
            "source_case_count": len(source_cases),
            "stream_count": len(streams),
            "external_inference_calls": 0,
        },
        "card": card,
        "cards": [card],
        "case_manifest": rel(card_manifest),
        "gates": gates,
        "usage_policy": {
            "private_pressure_only": True,
            "public_comparator_claims_allowed": False,
            "candidate_promotion_requires_real_benchmark_regression": True,
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if trigger_state == "GREEN" else 1


def load_source_cases(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in policy.get("input_manifests", []):
        path = resolve(str(item))
        for row in read_jsonl(path):
            if row.get("case_type") not in {"python_code_repair", "multi_stream_python_code_repair"}:
                continue
            if not row.get("buggy") or not row.get("tests"):
                continue
            rows.append({**row, "source_manifest": rel(path)})
    rows.sort(key=lambda row: (str(row.get("source_manifest")), str(row.get("case_id") or row.get("task_id"))))
    return rows


def build_multistream_case(case: dict[str, Any], streams: list[str], index: int) -> dict[str, Any]:
    case_id = f"multistream_{safe_name(case.get('case_id') or case.get('task_id') or index)}"
    task_id = str(case.get("task_id") or case.get("case_id") or case_id)
    tags = [str(tag) for tag in case.get("tags", []) if str(tag)] if isinstance(case.get("tags"), list) else []
    source_provenance = case.get("provenance") if isinstance(case.get("provenance"), dict) else {}
    rows = stream_rows(case_id, task_id, case, tags)
    return {
        "case_id": case_id,
        "case_type": "multi_stream_python_code_repair",
        "task_id": task_id,
        "signature": str(case.get("signature") or task_id),
        "buggy": str(case.get("buggy") or ""),
        "tests": str(case.get("tests") or ""),
        "tags": tags,
        "repair_templates": case.get("repair_templates") if isinstance(case.get("repair_templates"), dict) else {},
        "required_arms": [
            "benchmark_ratchet_arm",
            "code_repair_verifier",
            "residual_governance_arm",
            "monitorability_audit_arm",
            "planforge_critical_path_scheduler",
        ],
        "streams": streams,
        "stream_rows": rows,
        "causal_contract": {
            "strict_past_only": True,
            "idle_token": "-",
            "same_row_cross_stream_attention": "forbidden_in_verifier",
            "critical_path_scored": True,
        },
        "scoring": {
            "public_comparator_use": "forbidden",
            "score_semantics": "private_multistream_pressure_correctness_monitorability_and_critical_path",
        },
        "provenance": {
            "origin": "local_multi_stream_trace_factory",
            "source_manifest": case.get("source_manifest"),
            "source_case_id": case.get("case_id") or case.get("task_id"),
            "source_origin": source_provenance.get("origin"),
            "copied_public_benchmark_item_chars": 0,
            "external_inference_calls": 0,
            "concept_sources": [
                "arxiv:2605.12460",
                "arxiv:2510.17238",
                "arxiv:2512.07843",
                "arxiv:2504.15466",
            ],
        },
    }


def stream_rows(case_id: str, task_id: str, case: dict[str, Any], tags: list[str]) -> list[dict[str, Any]]:
    signature = str(case.get("signature") or task_id)
    tag_text = ",".join(tags) or "none"
    return [
        row(
            0,
            {
                "system_policy_stream": "local-only; no teacher; public comparator forbidden; keep streams causally ordered",
                "context_stream": f"task={task_id}; signature={signature}; residual_tags={tag_text}",
                "solver_stream": "-",
                "tool_test_stream": "-",
                "critic_audit_stream": "-",
                "patch_stream": "-",
                "residual_stream": "-",
                "visible_report_stream": "-",
            },
            {},
        ),
        row(
            1,
            {
                "system_policy_stream": "-",
                "context_stream": "buggy implementation and tests are now available to the local sandbox",
                "solver_stream": "propose first local candidate from buggy function",
                "tool_test_stream": "-",
                "critic_audit_stream": "audit likely failure mode from signature, tags, and tests",
                "patch_stream": "-",
                "residual_stream": "-",
                "visible_report_stream": "-",
            },
            {
                "context_stream": [["system_policy_stream", 0]],
                "solver_stream": [["context_stream", 0]],
                "critic_audit_stream": [["context_stream", 0]],
            },
        ),
        row(
            2,
            {
                "system_policy_stream": "-",
                "context_stream": "-",
                "solver_stream": "wait for sandbox and audit streams before retry",
                "tool_test_stream": "execute candidate in isolated Python tempdir",
                "critic_audit_stream": "classify stderr/stdout into residual class if tests fail",
                "patch_stream": "-",
                "residual_stream": "-",
                "visible_report_stream": "-",
            },
            {
                "solver_stream": [["solver_stream", 1], ["critic_audit_stream", 1]],
                "tool_test_stream": [["solver_stream", 1]],
                "critic_audit_stream": [["tool_test_stream", 1], ["solver_stream", 1]],
            },
        ),
        row(
            3,
            {
                "system_policy_stream": "-",
                "context_stream": "-",
                "solver_stream": "select repair candidate using transfer tags and audit result",
                "tool_test_stream": "execute repaired candidate in isolated Python tempdir",
                "critic_audit_stream": "compare repaired behavior against tests and hidden-residual pattern",
                "patch_stream": "emit patch candidate and patch trace hash",
                "residual_stream": "-",
                "visible_report_stream": "-",
            },
            {
                "solver_stream": [["tool_test_stream", 2], ["critic_audit_stream", 2]],
                "tool_test_stream": [["solver_stream", 2], ["critic_audit_stream", 2]],
                "critic_audit_stream": [["tool_test_stream", 2], ["critic_audit_stream", 2]],
                "patch_stream": [["solver_stream", 2], ["critic_audit_stream", 2]],
            },
        ),
        row(
            4,
            {
                "system_policy_stream": "-",
                "context_stream": "-",
                "solver_stream": "-",
                "tool_test_stream": "-",
                "critic_audit_stream": "monitor stream records whether audit caught the bug before final report",
                "patch_stream": "-",
                "residual_stream": "export residual or mastered-regression marker",
                "visible_report_stream": "-",
            },
            {
                "critic_audit_stream": [["tool_test_stream", 3], ["patch_stream", 3]],
                "residual_stream": [["critic_audit_stream", 3], ["tool_test_stream", 3], ["patch_stream", 3]],
            },
        ),
        row(
            5,
            {
                "system_policy_stream": "-",
                "context_stream": "-",
                "solver_stream": "-",
                "tool_test_stream": "-",
                "critic_audit_stream": "-",
                "patch_stream": "-",
                "residual_stream": "-",
                "visible_report_stream": f"report pass/fail, critical path, transfer consumption, and monitorability for {case_id}",
            },
            {
                "visible_report_stream": [["residual_stream", 4], ["critic_audit_stream", 4]],
            },
        ),
    ]


def row(index: int, cells: dict[str, str], dependencies: dict[str, list[list[Any]]]) -> dict[str, Any]:
    return {
        "row_index": index,
        "cells": {
            stream: {
                "text": text,
                "idle": text == "-",
                "depends_on": dependencies.get(stream, []),
                "token_estimate": estimate_tokens(text),
            }
            for stream, text in cells.items()
        },
    }


def build_card(policy: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    card = dict(policy.get("default_card") if isinstance(policy.get("default_card"), dict) else {})
    card_id = str(card.get("id") or "multistream_code_repair_pressure")
    case_manifest = str(card.get("case_manifest") or f"data/multi_stream_benchmarks/{card_id}.jsonl")
    card.update(
        {
            "schema": "sparkstream_benchmark_card_v0",
            "id": card_id,
            "source_id": card_id,
            "resource_pantry_path": case_manifest,
            "staged_path": case_manifest,
            "case_manifest": case_manifest,
            "public_comparator_use": "forbidden",
            "contamination_risk": "private_multistream_pressure_do_not_train_on_eval_cases",
            "permission_envelope": {
                "network": "forbidden_during_scoring",
                "external_inference": "forbidden",
                "hardware": "forbidden_without_explicit_human_approval",
            },
            "multi_stream_benchmark": {
                "mode": "code_repair_stream_table",
                "case_count": len(cases),
                "streams": policy.get("streams", []),
                "score_semantics": "private_multistream_pressure_correctness_monitorability_and_critical_path",
                "case_index": [
                    {
                        "case_id": case.get("case_id"),
                        "task_id": case.get("task_id"),
                        "source_case_id": case.get("provenance", {}).get("source_case_id"),
                        "required_arms": case.get("required_arms", []),
                    }
                    for case in cases
                ],
            },
            "external_inference_calls": 0,
        }
    )
    return card


def estimate_tokens(text: str) -> int:
    if not text or text == "-":
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    card = report.get("card", {})
    return "\n".join(
        [
            "# Multi-Stream Trace Factory",
            "",
            f"- Trigger: {report.get('trigger_state')}",
            f"- Cases: {summary.get('case_count')}",
            f"- Streams: {summary.get('stream_count')}",
            f"- Card: {card.get('id')}",
            f"- Manifest: {report.get('case_manifest')}",
            f"- External inference calls: {report.get('external_inference_calls')}",
            "",
            "Generated cases are private pressure only and cannot be used as public-comparator promotion evidence.",
            "",
        ]
    )


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "item")).strip("_") or "item"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
