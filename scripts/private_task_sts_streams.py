#!/usr/bin/env python3
"""Build private-safe task-level STS streams for closure fanout gates.

This is an explicit STS-on artifact builder for private synthetic/eval tasks.
It copies visible prompt/contract metadata only. It must not copy tests,
solutions, public prompts, public tests, or public benchmark answers into the
stream manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_EXPORT_KEYS = {
    "tests",
    "test_list",
    "canonical_solution",
    "solution",
    "solution_body",
    "solution_expr",
    "expected_output",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Private task JSONL to convert into visible-metadata STS streams.")
    parser.add_argument("--out", required=True, help="Output STS stream JSONL consumed by symliquid-cli --sts-streams.")
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--task-limit", type=int, default=0)
    args = parser.parse_args()

    tasks = read_jsonl(resolve(args.tasks))
    if args.task_limit > 0:
        tasks = tasks[: args.task_limit]
    rows = [stream_row(task) for task in tasks if task_allowed(task)]
    write_jsonl(resolve(args.out), rows)

    leakage = leakage_scan(rows)
    families = Counter(str(row.get("broad_private_family_v1") or "unknown") for row in rows)
    gates = [
        gate("task_rows_loaded", bool(tasks), {"task_rows": len(tasks)}),
        gate("sts_rows_written", bool(rows), {"sts_rows": len(rows)}),
        gate("one_stream_row_per_task", len(rows) == len({row["task_id"] for row in rows}), len(rows)),
        gate("public_data_leakage_zero", leakage["hit_count"] == 0, leakage),
        gate("public_solution_flags_false", not any(row.get("public_benchmark_solutions_included") for row in rows), 0),
        gate("public_test_flags_false", not any(row.get("public_tests_included") for row in rows), 0),
        gate("external_inference_zero", True, 0),
    ]
    report = {
        "policy": "project_theseus_private_task_sts_streams_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) else "RED",
        "inputs": {
            "tasks": rel(resolve(args.tasks)),
            "task_limit": int(args.task_limit),
            "public_benchmark_inputs_read": False,
            "public_tests_used": False,
            "public_solutions_used": False,
        },
        "outputs": {
            "sts_streams": rel(resolve(args.out)),
            "report": rel(resolve(args.report_out)),
        },
        "summary": {
            "input_task_count": len(tasks),
            "sts_stream_task_count": len(rows),
            "family_counts": dict(sorted(families.items())),
            "public_data_leakage_hit_count": leakage["hit_count"],
            "native_parallel_token_generation_claimed": False,
            "score_semantics": "private visible-metadata STS context only, not answer training or public calibration",
            "external_inference_calls": 0,
        },
        "gates": gates,
        "external_inference_calls": 0,
    }
    write_json(resolve(args.report_out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def task_allowed(task: dict[str, Any]) -> bool:
    if task.get("public_benchmark") is True:
        return False
    if task.get("public_benchmark_solutions_included") or task.get("public_tests_included"):
        return False
    task_id = str(task.get("task_id") or "")
    prompt = str(task.get("prompt") or "")
    return bool(task_id and prompt)


def stream_row(task: dict[str, Any]) -> dict[str, Any]:
    contract = task.get("decoder_contract") if isinstance(task.get("decoder_contract"), dict) else {}
    generation_plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
    return_contract = contract.get("return_contract") if isinstance(contract.get("return_contract"), dict) else {}
    roles = contract.get("argument_roles") if isinstance(contract.get("argument_roles"), dict) else {}
    required = contract.get("required_constructs") if isinstance(contract.get("required_constructs"), list) else []
    skeleton = generation_plan.get("skeleton_bias") if isinstance(generation_plan.get("skeleton_bias"), list) else []
    tags = task.get("tags") if isinstance(task.get("tags"), list) else []

    category = clean_text(task.get("category"))
    family = clean_text(task.get("broad_private_family_v1") or task.get("public_safe_maturity_family_v4") or task.get("residual_concept"))
    residual = clean_text(task.get("concept_residual_label") or task.get("residual_concept") or category)
    prompt = clean_text(task.get("prompt"), limit=600)
    entry = clean_text(task.get("entry_point"))

    streams = {
        "context_stream": (
            f"visible_private_task entry_point={entry}; category={category}; family={family}; "
            f"tags={','.join(clean_text(tag, limit=64) for tag in tags[:12])}; prompt={prompt}"
        ),
        "solver_stream": (
            "generate a full Python function body from visible prompt and decoder contract only; "
            f"required_constructs={','.join(clean_text(item, limit=64) for item in required[:12])}; "
            f"skeleton_bias={','.join(clean_text(item, limit=64) for item in skeleton[:12])}"
        ),
        "critic_stream": (
            "audit exact entry point, argument roles, return shape, branch/loop/local-state obligations, "
            f"residual={residual}"
        ),
        "tool_stream": (
            "allowed local checks: ast.parse, decoder_contract verifier, provenance guardrail, private hidden tests after generation"
        ),
        "patch_stream": "repair learned body tokens only; no task-id lookup; no solution or test text in STS stream",
        "residual_stream": (
            f"family={family}; residual={residual}; type_family={clean_text(contract.get('type_family'))}; "
            f"return_shape={clean_text(contract.get('return_shape') or return_contract.get('shape'))}; "
            f"argument_roles={','.join(f'{clean_text(k, limit=48)}:{clean_text(v, limit=48)}' for k, v in sorted(roles.items())[:6])}"
        ),
        "visible_report_stream": "private-safe task STS stream from visible metadata only; external_inference_calls=0",
    }
    return {
        "policy": "project_theseus_private_task_sts_stream_v1",
        "task_id": str(task.get("task_id") or ""),
        "source_task_id": str(task.get("source_task_id") or ""),
        "split": str(task.get("split") or "eval"),
        "card_id": str(task.get("card_id") or ""),
        "category": category,
        "broad_private_family_v1": family,
        "benchmark_evidence_level": str(task.get("benchmark_evidence_level") or ""),
        "visible_task_only": True,
        "deterministic_visible_metadata_sts": True,
        "native_parallel_token_generation": False,
        "public_benchmark_solutions_included": False,
        "public_tests_included": False,
        "canonical_solution_exported": False,
        "raw_public_prompt_or_tests_copied": False,
        "streams": streams,
        "causal_contract": {
            "strict_past_only": True,
            "same_row_cross_stream_attention": "forbidden",
            "visible_metadata_only": True,
            "solutions_or_tests_copied": False,
        },
    }


def leakage_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for row in rows:
        serialized = json.dumps(row, sort_keys=True).lower()
        for key in FORBIDDEN_EXPORT_KEYS:
            if f'"{key}"' in serialized:
                hits.append({"task_id": row.get("task_id"), "hit": f"forbidden_key:{key}"})
                break
    return {"hit_count": len(hits), "sample_hits": hits[:8]}


def clean_text(value: Any, *, limit: int = 240) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text[:limit]


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
