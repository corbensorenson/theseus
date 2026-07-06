"""Build local public-code benchmark manifests for multi-stream repair pressure.

The output is intentionally not a public benchmark score. It is a local,
network-free regression surface derived from staged benchmark loaders and repo
schemas so that the multi-stream critic/patch discipline can be tested on
real code benchmark infrastructure before it is allowed to influence candidate
promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STREAMS = [
    "system_policy_stream",
    "context_stream",
    "solver_stream",
    "tool_test_stream",
    "critic_audit_stream",
    "patch_stream",
    "residual_stream",
    "visible_report_stream",
]
REQUIRED_ARMS = [
    "benchmark_ratchet_arm",
    "code_repair_verifier",
    "residual_governance_arm",
    "monitorability_audit_arm",
    "planforge_critical_path_scheduler",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", required=True)
    parser.add_argument("--source-path", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--out", default="")
    parser.add_argument("--summary-out", default="")
    args = parser.parse_args()

    source_id = source_from_card(args.card_id)
    source_path = resolve(args.source_path) if args.source_path else default_source_path(args.card_id)
    out = resolve(args.out or f"data/public_code_benchmark_manifests/{safe_name(args.card_id)}_seed{args.seed}.jsonl")
    summary_out = resolve(
        args.summary_out
        or f"reports/public_code_benchmark_manifest_{safe_name(args.card_id)}_seed{args.seed}.json"
    )

    cases = build_cases(args.card_id, source_id, source_path)
    cases = rotate(cases, args.seed)[: max(1, int(args.max_cases))]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases), encoding="utf-8")

    evidence_files = sorted({file for case in cases for file in case.get("provenance", {}).get("source_evidence_files", [])})
    summary = {
        "policy": "project_theseus_public_code_benchmark_manifest_builder_v1",
        "created_utc": now(),
        "card_id": args.card_id,
        "source_id": source_id,
        "source_path": rel_or_abs(source_path),
        "manifest": rel_or_abs(out),
        "case_count": len(cases),
        "benchmark_evidence_level": "public_loader_regression",
        "public_benchmark_score_claim": "forbidden",
        "network_during_scoring": "forbidden",
        "external_inference_calls": 0,
        "source_evidence_files": evidence_files,
        "source_evidence_sha256": {
            path: sha256_file(resolve(path)) for path in evidence_files if resolve(path).exists()
        },
        "gates": [
            gate("source_present", source_path.exists(), rel_or_abs(source_path)),
            gate("cases_generated", bool(cases), f"cases={len(cases)}"),
            gate("public_comparator_quarantined", True, "public comparator use forbidden"),
            gate("external_inference_zero", True, "local manifest builder only"),
        ],
    }
    write_json(summary_out, summary)
    print(json.dumps(summary, indent=2))
    return 0 if cases else 1


def build_cases(card_id: str, source_id: str, source_path: Path) -> list[dict[str, Any]]:
    if not source_path.exists():
        return []
    if source_id in {"evalplus", "human_eval", "mbpp"}:
        return evalplus_cases(card_id, source_id, source_path)
    if source_id == "bigcodebench":
        return bigcodebench_cases(card_id, source_path)
    if source_id == "livecodebench":
        return livecodebench_cases(card_id, source_path)
    return generic_code_loader_cases(card_id, source_id, source_path)


def evalplus_cases(card_id: str, source_id: str, source_path: Path) -> list[dict[str, Any]]:
    evidence = existing_relpaths(
        source_path,
        [
            "evalplus/data/humaneval.py",
            "evalplus/data/mbpp.py",
            "evalplus/data/utils.py",
            "evalplus/evaluate.py",
        ],
    )
    tags = ["parsing", "type_handling", "benchmark_loader", "repair_loop"]
    return [
        case(
            card_id,
            source_id,
            "extract_entry_point",
            "extract_entry_point(prompt)",
            "def extract_entry_point(prompt):\n    return ''\n",
            "assert extract_entry_point('def add(a, b):\\n    return a + b\\n') == 'add'\n"
            "assert extract_entry_point('import math\\n\\ndef solve(nums):\\n    pass\\n') == 'solve'\n",
            tags,
            evidence,
            "EvalPlus/HumanEval loader contracts identify tasks by entry_point extracted from Python prompts.",
        ),
        case(
            card_id,
            source_id,
            "has_required_keys",
            "has_required_keys(row, keys)",
            "def has_required_keys(row, keys):\n    return True\n",
            "assert has_required_keys({'task_id':'HumanEval/0','prompt':'p','test':'t','entry_point':'add'}, ['task_id','prompt','test','entry_point']) is True\n"
            "assert has_required_keys({'task_id':'HumanEval/0','prompt':'p'}, ['task_id','prompt','test','entry_point']) is False\n",
            tags,
            evidence,
            "EvalPlus rows must expose task_id, prompt, test, and entry_point before sandbox execution.",
        ),
        case(
            card_id,
            source_id,
            "parse_ints",
            "parse_ints(text)",
            "def parse_ints(text):\n    return [int(part) for part in text.split(',')]\n",
            "assert parse_ints('1, 2, x, -3') == [1, 2, -3]\nassert parse_ints('') == []\n",
            tags,
            evidence,
            "EvalPlus adapters repeatedly parse compact numeric metadata and version fragments while keeping bad tokens harmless.",
        ),
    ]


def bigcodebench_cases(card_id: str, source_path: Path) -> list[dict[str, Any]]:
    evidence = existing_relpaths(
        source_path,
        [
            "bigcodebench/data/bigcodebench.py",
            "bigcodebench/evaluate.py",
            "analysis/task2domain.json",
            "analysis/lib2domain.json",
        ],
    )
    tags = ["parsing", "algorithm_choice", "benchmark_loader", "repair_loop"]
    return [
        case(
            card_id,
            "bigcodebench",
            "has_required_keys",
            "has_required_keys(row, keys)",
            "def has_required_keys(row, keys):\n    return True\n",
            "required = ['task_id','complete_prompt','instruct_prompt','canonical_solution','test','entry_point']\n"
            "assert has_required_keys({'task_id':'BigCodeBench/1','complete_prompt':'c','instruct_prompt':'i','canonical_solution':'s','test':'t','entry_point':'solve'}, required) is True\n"
            "assert has_required_keys({'task_id':'BigCodeBench/1','complete_prompt':'c'}, required) is False\n",
            tags,
            evidence,
            "BigCodeBench loader rows have a fixed code-generation schema before any generated solution is scored.",
        ),
        case(
            card_id,
            "bigcodebench",
            "stable_dedupe",
            "stable_dedupe(xs)",
            "def stable_dedupe(xs):\n    return list(set(xs))\n",
            "assert stable_dedupe(['io', 'math', 'io', 'json']) == ['io', 'math', 'json']\nassert stable_dedupe([]) == []\n",
            tags,
            evidence,
            "BigCodeBench analysis metadata maps tasks and libraries to domains; preserving order matters for reproducible routing.",
        ),
        case(
            card_id,
            "bigcodebench",
            "extract_entry_point",
            "extract_entry_point(prompt)",
            "def extract_entry_point(prompt):\n    return ''\n",
            "assert extract_entry_point('def solve_grid(grid):\\n    pass\\n') == 'solve_grid'\n"
            "assert extract_entry_point('from typing import List\\n\\ndef transform(x):\\n    return x\\n') == 'transform'\n",
            tags,
            evidence,
            "BigCodeBench scoring depends on matching generated functions to the entry_point.",
        ),
    ]


def livecodebench_cases(card_id: str, source_path: Path) -> list[dict[str, Any]]:
    evidence = existing_relpaths(
        source_path,
        [
            "lcb_runner/benchmarks/code_generation.py",
            "lcb_runner/evaluation/testing_util.py",
            "lcb_runner/prompts/few_shot_examples/generation/func.json",
            "lcb_runner/prompts/few_shot_examples/generation/stdin.json",
        ],
    )
    tags = ["edge_case", "type_handling", "benchmark_loader", "repair_loop"]
    return [
        case(
            card_id,
            "livecodebench",
            "count_public_tests",
            "count_public_tests(problem)",
            "def count_public_tests(problem):\n    return 0\n",
            "assert count_public_tests({'public_test_cases':[1, 2], 'private_test_cases':[3]}) == 2\n"
            "assert count_public_tests({'metadata': {}}) == 0\n",
            tags,
            evidence,
            "LiveCodeBench separates public and private tests; pressure must count public tests without peeking at private cases.",
        ),
        case(
            card_id,
            "livecodebench",
            "safe_head",
            "safe_head(xs, default=None)",
            "def safe_head(xs, default=None):\n    return xs[0]\n",
            "assert safe_head([], 'no_public_case') == 'no_public_case'\nassert safe_head(['sample'], 'no_public_case') == 'sample'\n",
            tags,
            evidence,
            "LiveCodeBench adapters must handle empty public-test lists and sparse metadata without crashing.",
        ),
        case(
            card_id,
            "livecodebench",
            "normalize_test_type",
            "normalize_test_type(value)",
            "def normalize_test_type(value):\n    return value\n",
            "assert normalize_test_type('STDIN') == 'stdin'\nassert normalize_test_type('functional') == 'functional'\nassert normalize_test_type('weird') == 'unknown'\n",
            tags,
            evidence,
            "LiveCodeBench test cases route through stdin versus functional modes.",
        ),
    ]


def generic_code_loader_cases(card_id: str, source_id: str, source_path: Path) -> list[dict[str, Any]]:
    evidence = [rel_or_abs(path) for path in list(source_path.glob("*.toml"))[:2] + list(source_path.glob("*.md"))[:2]]
    return [
        case(
            card_id,
            source_id,
            "has_required_keys",
            "has_required_keys(row, keys)",
            "def has_required_keys(row, keys):\n    return True\n",
            "assert has_required_keys({'prompt':'p','tests':'t'}, ['prompt','tests']) is True\n"
            "assert has_required_keys({'prompt':'p'}, ['prompt','tests']) is False\n",
            ["type_handling", "benchmark_loader", "repair_loop"],
            evidence,
            "Generic code benchmark loaders must validate rows before sandboxed scoring.",
        )
    ]


def case(
    card_id: str,
    source_id: str,
    slug: str,
    signature: str,
    buggy: str,
    tests: str,
    tags: list[str],
    evidence_files: list[str],
    rationale: str,
) -> dict[str, Any]:
    task_id = f"{safe_name(card_id)}_{slug}"
    return {
        "case_id": f"public_loader_{task_id}",
        "case_type": "multi_stream_python_code_repair",
        "task_id": task_id,
        "signature": signature,
        "buggy": buggy,
        "tests": tests,
        "tags": tags,
        "repair_templates": {},
        "streams": STREAMS,
        "stream_rows": stream_rows(task_id, signature, tags, source_id),
        "causal_contract": {
            "strict_past_only": True,
            "same_row_cross_stream_attention": "forbidden_in_verifier",
            "critical_path_scored": True,
            "idle_token": "-",
        },
        "provenance": {
            "origin": "local_public_code_benchmark_manifest_builder",
            "benchmark_evidence_level": "public_loader_regression",
            "source_card_id": card_id,
            "source_id": source_id,
            "source_evidence_files": evidence_files,
            "source_schema_rationale": rationale,
            "copied_public_benchmark_item_chars": 0,
            "external_inference_calls": 0,
        },
        "scoring": {
            "public_comparator_use": "forbidden",
            "score_semantics": "public_loader_regression_not_benchmark_score",
            "benchmark_score_claim": "forbidden",
        },
        "required_arms": REQUIRED_ARMS,
    }


def stream_rows(task_id: str, signature: str, tags: list[str], source_id: str) -> list[dict[str, Any]]:
    context = f"public-loader-regression source={source_id}; task={task_id}; signature={signature}; tags={','.join(tags)}"
    return [
        row(
            0,
            {
                "system_policy_stream": "local-only; no teacher; public comparator forbidden; public loader regression only",
                "context_stream": context,
            },
        ),
        row(
            1,
            {
                "context_stream": "buggy adapter helper and local unit tests are available to the sandbox",
                "solver_stream": "emit first local candidate from buggy function",
                "critic_audit_stream": "audit likely benchmark-loader failure mode from signature and tests",
            },
            deps={
                "context_stream": [["system_policy_stream", 0]],
                "solver_stream": [["context_stream", 0]],
                "critic_audit_stream": [["context_stream", 0]],
            },
        ),
        row(
            2,
            {
                "tool_test_stream": "execute candidate in isolated Python tempdir",
                "critic_audit_stream": "classify stderr/stdout into residual class if tests fail",
                "solver_stream": "wait for sandbox and audit streams before retry",
            },
            deps={
                "tool_test_stream": [["solver_stream", 1]],
                "critic_audit_stream": [["tool_test_stream", 1], ["solver_stream", 1]],
                "solver_stream": [["solver_stream", 1], ["critic_audit_stream", 1]],
            },
        ),
        row(
            3,
            {
                "solver_stream": "select repair candidate using audit result and transfer categories",
                "tool_test_stream": "execute repaired candidate in isolated Python tempdir",
                "critic_audit_stream": "compare repaired behavior against tests and residual pattern",
                "patch_stream": "emit bounded patch candidate and patch trace hash",
            },
            deps={
                "solver_stream": [["tool_test_stream", 2], ["critic_audit_stream", 2]],
                "tool_test_stream": [["solver_stream", 2], ["critic_audit_stream", 2]],
                "critic_audit_stream": [["tool_test_stream", 2], ["critic_audit_stream", 2]],
                "patch_stream": [["solver_stream", 2], ["critic_audit_stream", 2]],
            },
        ),
        row(
            4,
            {
                "critic_audit_stream": "monitor stream records whether audit caught bug before final report",
                "residual_stream": "export residual or mastered-regression marker",
            },
            deps={
                "critic_audit_stream": [["tool_test_stream", 3], ["patch_stream", 3]],
                "residual_stream": [["critic_audit_stream", 3], ["tool_test_stream", 3], ["patch_stream", 3]],
            },
        ),
        row(
            5,
            {
                "visible_report_stream": f"report public-loader regression pass/fail, critical path, and transfer consumption for {task_id}",
            },
            deps={"visible_report_stream": [["residual_stream", 4], ["critic_audit_stream", 4]]},
        ),
    ]


def row(index: int, values: dict[str, str], deps: dict[str, list[list[Any]]] | None = None) -> dict[str, Any]:
    deps = deps or {}
    cells = {}
    for stream in STREAMS:
        text = values.get(stream, "-")
        cells[stream] = {
            "text": text,
            "idle": text == "-",
            "depends_on": deps.get(stream, []),
            "token_estimate": estimate_tokens(text) if text != "-" else 0,
        }
    return {"row_index": index, "cells": cells}


def existing_relpaths(source_path: Path, relpaths: list[str]) -> list[str]:
    out = []
    for item in relpaths:
        path = source_path / item
        if path.exists():
            out.append(rel_or_abs(path))
    return out


def default_source_path(card_id: str) -> Path:
    card = read_json(ROOT / "benchmarks" / "cards" / f"{card_id}.json", {})
    raw = card.get("resource_pantry_path") or card.get("staged_path") or ""
    return resolve(str(raw)) if raw else ROOT


def source_from_card(card_id: str) -> str:
    card = read_json(ROOT / "benchmarks" / "cards" / f"{card_id}.json", {})
    return str(card.get("source_id") or card.get("id") or card_id).replace("source_", "")


def rotate(cases: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    if not cases:
        return []
    offset = seed % len(cases)
    return cases[offset:] + cases[:offset]


def estimate_tokens(text: str) -> int:
    if not text or text == "-":
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel_or_abs(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "item")).strip("_").lower() or "item"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
