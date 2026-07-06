"""Stream Token Superposition (STS) learning forge.

This creates governed, trainable multi-stream traces for Theseus without
overclaiming architecture proof. It is a dataset/contract lane: context,
solver, critic, tool, patch, residual, and report streams are aligned in
causal rows so a future SymLiquid decoder can learn to emit multiple streams.

No public benchmark solutions are admitted. Public code calibration remains
evaluation-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STREAMS = [
    "context_stream",
    "solver_stream",
    "critic_stream",
    "tool_stream",
    "patch_stream",
    "residual_stream",
    "visible_report_stream",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-curriculum", default="data/private_code_curriculum/code_lm_closure_seed14.jsonl")
    parser.add_argument("--multi-stream-manifest", default="data/multi_stream_benchmarks/multistream_code_repair_pressure.jsonl")
    parser.add_argument("--max-private-rows", type=int, default=720)
    parser.add_argument("--out-data", default="data/sts_learning/sts_code_streams_seed14.jsonl")
    parser.add_argument("--out", default="reports/sts_learning_forge.json")
    args = parser.parse_args()

    private_rows = read_jsonl(resolve(args.private_curriculum))[: max(0, args.max_private_rows)]
    private_cases = read_jsonl(resolve(args.multi_stream_manifest))
    rows = []
    for index, task in enumerate(private_rows):
        if not private_task_allowed(task):
            continue
        rows.extend(code_task_stream_rows(task, index))
    for index, case in enumerate(private_cases):
        if private_multistream_case_allowed(case):
            rows.extend(multistream_case_rows(case, index))

    train_rows = [row for row in rows if row.get("split") == "train"]
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    write_jsonl(resolve(args.out_data), rows)

    gates = [
        gate("private_curriculum_loaded", bool(private_rows), f"rows={len(private_rows)}"),
        gate("private_stream_rows_written", bool(rows), f"rows={len(rows)}"),
        gate("train_eval_split_present", bool(train_rows) and bool(eval_rows), f"train={len(train_rows)} eval={len(eval_rows)}"),
        gate("stream_schema_parallel", all(set(DEFAULT_STREAMS).issubset(set(row.get("streams", []))) for row in rows), f"streams={len(DEFAULT_STREAMS)}"),
        gate("independent_output_streams_present", output_stream_count(rows) >= 4, f"output_streams={output_stream_count(rows)}"),
        gate("no_public_benchmark_rows", not any(row.get("public_benchmark") for row in rows), "public_benchmark=False"),
        gate("no_public_benchmark_solutions", not any("public_benchmark" in str(row.get("source_id", "")).lower() for row in rows), "source ids private/local only"),
        gate("native_parallel_token_generation_not_overclaimed", True, "this is STS train/eval substrate, not yet native decoder proof"),
        gate("external_inference_zero", True, "deterministic local trace construction"),
    ]
    trigger_state = "GREEN" if all(item["passed"] for item in gates) else "YELLOW"
    report = {
        "policy": "project_theseus_sts_learning_forge_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "methodology": "private_code_to_stream_token_superposition_training_traces",
        "streams": DEFAULT_STREAMS,
        "artifacts": {
            "sts_train_eval_jsonl": rel(resolve(args.out_data)),
            "private_curriculum": rel(resolve(args.private_curriculum)),
            "multi_stream_manifest": rel(resolve(args.multi_stream_manifest)),
        },
        "summary": {
            "row_count": len(rows),
            "train_row_count": len(train_rows),
            "eval_row_count": len(eval_rows),
            "private_code_task_count": len([row for row in private_rows if private_task_allowed(row)]),
            "private_multistream_case_count": len([row for row in private_cases if private_multistream_case_allowed(row)]),
            "stream_count": len(DEFAULT_STREAMS),
            "independent_output_stream_count": output_stream_count(rows),
            "native_parallel_token_generation_proven": False,
            "sts_training_substrate_ready": bool(rows) and bool(train_rows) and bool(eval_rows),
            "public_benchmark_solutions_included": False,
            "external_inference_calls": 0,
        },
        "next_integration": {
            "required": "Train a SymLiquid/Rust decoder to emit one token per output stream per causal row.",
            "promotion_claim_allowed_now": False,
            "why": "The forge proves governed STS data readiness, not learned parallel-stream generation yet.",
        },
        "gates": gates,
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if trigger_state == "GREEN" else 1


def code_task_stream_rows(task: dict[str, Any], index: int) -> list[dict[str, Any]]:
    split = "eval" if str(task.get("split") or "") == "eval" else "train"
    task_id = str(task.get("task_id") or f"private_task_{index}")
    prompt = str(task.get("prompt") or "")
    expr = str(task.get("solution_expr") or "")
    category = str(task.get("category") or "")
    entry = str(task.get("entry_point") or "solution")
    tests_hash = sha256_text(str(task.get("tests") or ""))[:16]
    base = {
        "policy": "project_theseus_sts_learning_row_v1",
        "source_id": str(task.get("source_id") or "local_generated_private_code_curriculum"),
        "task_id": f"sts_{task_id}",
        "split": split,
        "streams": DEFAULT_STREAMS,
        "public_benchmark": False,
        "benchmark_evidence_level": "private_sts_train_or_eval_only",
        "causal_contract": {
            "strict_past_only": True,
            "same_row_cross_stream_attention": "forbidden",
            "one_token_per_output_stream_target": True,
        },
    }
    return [
        {
            **base,
            "row_index": 0,
            "input_streams": {"context_stream": prompt},
            "target_streams": {
                "solver_stream": f"plan category={category}; entry={entry}",
                "critic_stream": "identify data shape and edge cases before code",
                "tool_stream": "sandbox not executed yet",
                "patch_stream": "-",
                "residual_stream": "-",
                "visible_report_stream": "-",
            },
        },
        {
            **base,
            "row_index": 1,
            "input_streams": {"context_stream": prompt, "solver_stream": "emit candidate expression"},
            "target_streams": {
                "solver_stream": expr,
                "critic_stream": expression_features(expr),
                "tool_stream": f"private_tests_sha={tests_hash}",
                "patch_stream": f"return {expr}",
                "residual_stream": "private_expected_pass_after_training",
                "visible_report_stream": "candidate generated from private curriculum only",
            },
        },
    ]


def multistream_case_rows(case: dict[str, Any], index: int) -> list[dict[str, Any]]:
    task_id = str(case.get("task_id") or case.get("case_id") or f"case_{index}")
    tags = ",".join(str(tag) for tag in case.get("tags", []) if str(tag)) if isinstance(case.get("tags"), list) else ""
    source_id = f"private_multistream:{case.get('case_id') or index}"
    return [
        {
            "policy": "project_theseus_sts_learning_row_v1",
            "source_id": source_id,
            "task_id": f"sts_multistream_{task_id}",
            "split": "train",
            "row_index": 0,
            "streams": DEFAULT_STREAMS,
            "public_benchmark": False,
            "benchmark_evidence_level": "private_sts_train_or_eval_only",
            "causal_contract": {
                "strict_past_only": True,
                "same_row_cross_stream_attention": "forbidden",
                "one_token_per_output_stream_target": True,
            },
            "input_streams": {
                "context_stream": f"repair task={task_id}; tags={tags}",
                "solver_stream": "candidate repair required",
            },
            "target_streams": {
                "solver_stream": "read buggy code and propose repair",
                "critic_stream": "classify likely residual before test",
                "tool_stream": "execute candidate in sandbox",
                "patch_stream": "emit minimal patch",
                "residual_stream": "export residual if tests fail",
                "visible_report_stream": "record stream provenance",
            },
        }
    ]


def private_task_allowed(task: dict[str, Any]) -> bool:
    return bool(
        task.get("public_benchmark") is False
        and str(task.get("benchmark_evidence_level") or "").startswith(("private_", "permissive_open_source"))
        and task.get("solution_expr")
    )


def private_multistream_case_allowed(case: dict[str, Any]) -> bool:
    scoring = case.get("scoring") if isinstance(case.get("scoring"), dict) else {}
    return str(scoring.get("public_comparator_use") or "") == "forbidden" and bool(case.get("streams"))


def expression_features(expr: str) -> str:
    features = []
    for token in ["sum", "len", "sorted", "range", "set", "dict", "min", "max", "zip", "lambda"]:
        if token in expr:
            features.append(token)
    return "features=" + ",".join(features or ["literal_or_indexing"])


def output_stream_count(rows: list[dict[str, Any]]) -> int:
    streams = set()
    for row in rows:
        targets = row.get("target_streams") if isinstance(row.get("target_streams"), dict) else {}
        for key, value in targets.items():
            if value not in {"", "-"}:
                streams.add(key)
    return len(streams)


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


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


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
