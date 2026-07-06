#!/usr/bin/env python3
"""Materialize private residual pressure for the STS-default Code LM path.

This script reads only aggregate residual family names/counts from the public
transfer residual packet. It never copies public benchmark prompts, tests,
solutions, answers, or candidate code into training rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_rows import STS_DEFAULT_RESIDUAL_PRESSURE_PRIVATE_ROWS  # noqa: E402

DEFAULT_RESIDUAL_PACKET = ROOT / "reports" / "public_transfer_residual_packet.json"
DEFAULT_OUT = ROOT / "reports" / "sts_default_residual_pressure.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "sts_default_residual_pressure.md"
DEFAULT_JSONL = Path(STS_DEFAULT_RESIDUAL_PRESSURE_PRIVATE_ROWS[0])
TARGET_FAMILIES = ("algorithm_choice", "local_code_generation_adapter_needed")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def dominant_residual_counts(packet: dict[str, Any]) -> dict[str, int]:
    pairs = packet.get("dominant_residuals")
    if not isinstance(pairs, list):
        pairs = get_path(packet, ["summary", "dominant_residuals"], [])
    out: dict[str, int] = {}
    if isinstance(pairs, list):
        for item in pairs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    out[str(item[0])] = int(item[1])
                except Exception:
                    continue
            elif isinstance(item, dict):
                family = str(item.get("family") or item.get("residual") or "")
                if family:
                    out[family] = int(item.get("count") or 0)
    return out


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def decoder_contract(
    *,
    category: str,
    return_shape: str,
    arg_count: int,
    required_constructs: list[str],
    skeleton_bias: list[str],
) -> dict[str, Any]:
    return {
        "policy": "project_theseus_decoder_contract_v1",
        "category": category,
        "type_family": category,
        "return_shape": return_shape,
        "full_body_required": True,
        "guardrail_only": True,
        "public_tests_used": False,
        "public_solutions_used": False,
        "score_semantics": "private residual contract for routing and decoding pressure, not benchmark evidence",
        "visible_arg_count_hint": arg_count,
        "required_constructs": required_constructs,
        "residual_label_hint": category,
        "return_contract": {
            "shape": return_shape,
            "must_preserve_container_shape": return_shape in {"list", "dict", "tuple"},
            "empty_or_invalid_behavior": "infer_from_private_prompt",
            "source": "private_generated_contract_metadata_only",
        },
        "generation_plan": {
            "policy": "signature -> argument_roles -> algorithm_choice -> state_variables -> branch_loop_skeleton -> body -> repair",
            "repair_strategy": "choose a concrete algorithm before token-level decoding and keep local variable names aligned to the prompt",
            "skeleton_bias": skeleton_bias,
            "verifier_feedback": [
                "visible_argument_mismatch",
                "return_shape_mismatch",
                "missing_required_algorithm",
                "missing_required_skeleton",
                "semantic_family_mismatch",
                "semantic_admissibility_rejected",
            ],
            "public_tests_used": False,
            "public_solutions_used": False,
        },
    }


def algorithm_templates() -> list[dict[str, Any]]:
    return [
        {
            "name": "two_sum_first_pair",
            "prompt": "Return the first pair of indices whose values sum to the target, or (-1, -1) when no pair exists.",
            "return_shape": "tuple",
            "arg_count": 2,
            "body": "seen = {}\nfor idx, value in enumerate(data):\n    need = other - value\n    if need in seen:\n        return (seen[need], idx)\n    if value not in seen:\n        seen[value] = idx\nreturn (-1, -1)",
            "required": ["algorithm_choice", "hash_map", "loop", "branch", "locals"],
            "bias": ["hash_map_lookup", "first_valid_pair", "early_return_guard"],
        },
        {
            "name": "interval_union_length",
            "prompt": "Return the total covered length of half-open intervals after merging overlaps.",
            "return_shape": "int",
            "arg_count": 1,
            "body": "intervals = sorted((int(left), int(right)) for left, right in data if right > left)\nif not intervals:\n    return 0\ntotal = 0\ncur_left, cur_right = intervals[0]\nfor left, right in intervals[1:]:\n    if left <= cur_right:\n        cur_right = max(cur_right, right)\n    else:\n        total += cur_right - cur_left\n        cur_left, cur_right = left, right\nreturn total + cur_right - cur_left",
            "required": ["algorithm_choice", "sort", "loop", "branch", "interval_merge"],
            "bias": ["sort_then_merge", "stateful_interval_scan", "empty_input_guard"],
        },
        {
            "name": "grid_shortest_path",
            "prompt": "Return the shortest 4-neighbor path length across a 0/1 grid from top-left to bottom-right, or -1 if blocked.",
            "return_shape": "int",
            "arg_count": 1,
            "body": "from collections import deque\nif not data or not data[0] or data[0][0] or data[-1][-1]:\n    return -1\nrows, cols = len(data), len(data[0])\nqueue = deque([(0, 0, 1)])\nseen = {(0, 0)}\nwhile queue:\n    row, col, dist = queue.popleft()\n    if row == rows - 1 and col == cols - 1:\n        return dist\n    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):\n        nr, nc = row + dr, col + dc\n        if 0 <= nr < rows and 0 <= nc < cols and not data[nr][nc] and (nr, nc) not in seen:\n            seen.add((nr, nc))\n            queue.append((nr, nc, dist + 1))\nreturn -1",
            "required": ["algorithm_choice", "bfs", "queue", "loop", "branch"],
            "bias": ["bfs_queue", "visited_set", "grid_bounds_guard"],
        },
        {
            "name": "top_k_frequent",
            "prompt": "Return the k most frequent items, sorting ties by their string representation.",
            "return_shape": "list",
            "arg_count": 2,
            "body": "from collections import Counter\ncounts = Counter(data)\nranked = sorted(counts, key=lambda item: (-counts[item], str(item)))\nreturn ranked[:max(0, other)]",
            "required": ["algorithm_choice", "frequency_map", "sort", "locals"],
            "bias": ["counter_frequency", "tie_breaker", "bounded_slice"],
        },
        {
            "name": "longest_valid_parentheses",
            "prompt": "Return the length of the longest valid parentheses substring.",
            "return_shape": "int",
            "arg_count": 1,
            "body": "stack = [-1]\nbest = 0\nfor idx, ch in enumerate(data):\n    if ch == '(':\n        stack.append(idx)\n    else:\n        stack.pop()\n        if stack:\n            best = max(best, idx - stack[-1])\n        else:\n            stack.append(idx)\nreturn best",
            "required": ["algorithm_choice", "stack", "loop", "branch", "locals"],
            "bias": ["sentinel_stack", "span_length_update", "invalid_close_reset"],
        },
        {
            "name": "weighted_interval_schedule",
            "prompt": "Return the maximum value from non-overlapping jobs represented as (start, end, value).",
            "return_shape": "int",
            "arg_count": 1,
            "body": "import bisect\njobs = sorted((int(s), int(e), int(v)) for s, e, v in data if e >= s)\nends = [job[1] for job in jobs]\ndp = [0]\nfor idx, (start, end, value) in enumerate(jobs, 1):\n    prev = bisect.bisect_right(ends, start, 0, idx - 1)\n    dp.append(max(dp[-1], dp[prev] + value))\nreturn dp[-1]",
            "required": ["algorithm_choice", "dynamic_programming", "binary_search", "sort", "loop"],
            "bias": ["sort_by_end", "bisect_previous_compatible", "dp_table"],
        },
        {
            "name": "small_knapsack",
            "prompt": "Return the best value for items represented as (weight, value) under a capacity limit.",
            "return_shape": "int",
            "arg_count": 2,
            "body": "capacity = max(0, int(other))\ndp = [0] * (capacity + 1)\nfor weight, value in data:\n    weight, value = int(weight), int(value)\n    if weight <= 0:\n        continue\n    for cap in range(capacity, weight - 1, -1):\n        dp[cap] = max(dp[cap], dp[cap - weight] + value)\nreturn dp[capacity]",
            "required": ["algorithm_choice", "dynamic_programming", "reverse_loop", "locals"],
            "bias": ["capacity_dp", "reverse_capacity_loop", "invalid_weight_guard"],
        },
        {
            "name": "component_sizes",
            "prompt": "Return sorted connected-component sizes from an undirected adjacency dictionary.",
            "return_shape": "list",
            "arg_count": 1,
            "body": "seen = set()\nsizes = []\nfor node in data:\n    if node in seen:\n        continue\n    stack = [node]\n    seen.add(node)\n    size = 0\n    while stack:\n        cur = stack.pop()\n        size += 1\n        for nxt in data.get(cur, []):\n            if nxt not in seen:\n                seen.add(nxt)\n                stack.append(nxt)\n    sizes.append(size)\nreturn sorted(sizes)",
            "required": ["algorithm_choice", "graph_traversal", "stack", "loop", "branch"],
            "bias": ["dfs_stack", "visited_set", "component_counter"],
        },
    ]


def adapter_templates() -> list[dict[str, Any]]:
    return [
        {
            "name": "records_group_totals",
            "prompt": "Given records with 'group' and 'value', return a dictionary of numeric totals per group.",
            "return_shape": "dict",
            "arg_count": 1,
            "body": "out = {}\nfor record in data or []:\n    if not isinstance(record, dict):\n        continue\n    group = str(record.get('group', ''))\n    try:\n        value = float(record.get('value', 0))\n    except Exception:\n        value = 0.0\n    out[group] = out.get(group, 0.0) + value\nreturn out",
            "required": ["local_adapter", "dict_return_builder", "loop", "branch", "coercion"],
            "bias": ["named_local_state", "safe_record_access", "dict_accumulator"],
        },
        {
            "name": "safe_nested_lookup",
            "prompt": "Return a nested dictionary value for a path of keys, or the fallback when any level is missing.",
            "return_shape": "unknown",
            "arg_count": 2,
            "body": "current = data\npath, fallback = other\nfor key in path:\n    if not isinstance(current, dict) or key not in current:\n        return fallback\n    current = current[key]\nreturn current",
            "required": ["local_adapter", "loop", "branch", "nested_lookup"],
            "bias": ["guarded_nested_access", "fallback_return", "path_iteration"],
        },
        {
            "name": "parse_key_value_lines",
            "prompt": "Parse newline-separated key=value lines into a dictionary, ignoring malformed lines.",
            "return_shape": "dict",
            "arg_count": 1,
            "body": "out = {}\nfor line in str(data).splitlines():\n    if '=' not in line:\n        continue\n    key, value = line.split('=', 1)\n    key = key.strip()\n    if key:\n        out[key] = value.strip()\nreturn out",
            "required": ["local_adapter", "string_parse", "dict_return_builder", "loop", "branch"],
            "bias": ["split_once", "strip_fields", "malformed_line_guard"],
        },
        {
            "name": "matrix_shape",
            "prompt": "Return (row_count, max_column_count, rectangular) for a nested list matrix.",
            "return_shape": "tuple",
            "arg_count": 1,
            "body": "rows = data if isinstance(data, list) else []\nlengths = [len(row) if isinstance(row, list) else 0 for row in rows]\nmax_cols = max(lengths, default=0)\nrectangular = all(length == max_cols for length in lengths)\nreturn (len(rows), max_cols, rectangular)",
            "required": ["local_adapter", "return_shape_tuple", "locals", "container_shape"],
            "bias": ["shape_summary", "default_for_empty", "bool_contract"],
        },
        {
            "name": "sort_record_names",
            "prompt": "Return item names from records sorted by priority descending, then name ascending.",
            "return_shape": "list",
            "arg_count": 1,
            "body": "records = [row for row in data or [] if isinstance(row, dict)]\nrecords.sort(key=lambda row: (-int(row.get('priority', 0)), str(row.get('name', ''))))\nreturn [str(row.get('name', '')) for row in records]",
            "required": ["local_adapter", "sort", "list_return_builder", "coercion"],
            "bias": ["record_filter", "multi_key_sort", "field_projection"],
        },
        {
            "name": "coerce_bool_flags",
            "prompt": "Normalize a mapping of string flags into booleans using yes/no/on/off/1/0 values.",
            "return_shape": "dict",
            "arg_count": 1,
            "body": "truthy = {'1', 'true', 'yes', 'on'}\nfalsy = {'0', 'false', 'no', 'off'}\nout = {}\nfor key, value in dict(data or {}).items():\n    text = str(value).strip().lower()\n    if text in truthy:\n        out[key] = True\n    elif text in falsy:\n        out[key] = False\nreturn out",
            "required": ["local_adapter", "dict_return_builder", "branch", "coercion"],
            "bias": ["truth_table_sets", "string_normalization", "skip_unknown_values"],
        },
        {
            "name": "flatten_tag_counts",
            "prompt": "Return counts for tags inside records where each record may hold a list under 'tags'.",
            "return_shape": "dict",
            "arg_count": 1,
            "body": "out = {}\nfor record in data or []:\n    tags = record.get('tags', []) if isinstance(record, dict) else []\n    for tag in (tags if isinstance(tags, list) else []):\n        key = str(tag)\n        out[key] = out.get(key, 0) + 1\nreturn out",
            "required": ["local_adapter", "nested_loop", "dict_return_builder", "branch"],
            "bias": ["nested_collection_guard", "counter_dict", "record_schema_adapter"],
        },
        {
            "name": "bounded_pages",
            "prompt": "Return page slices from a list using one-based page number and page size.",
            "return_shape": "list",
            "arg_count": 2,
            "body": "page, size = other\npage = max(1, int(page))\nsize = max(1, int(size))\nstart = (page - 1) * size\nreturn list(data or [])[start:start + size]",
            "required": ["local_adapter", "index_math", "slice", "locals"],
            "bias": ["one_based_page_math", "bounds_normalization", "list_slice_return"],
        },
    ]


def build_row(
    *,
    family: str,
    idx: int,
    template: dict[str, Any],
    source_count: int,
    out_path: Path,
    residual_packet_path: Path,
) -> dict[str, Any]:
    task_id = f"sts_default_{family}_{template['name']}_{idx:04d}"
    return {
        "task_id": task_id,
        "source_task_id": task_id,
        "card_id": "private_sts_default_residual_pressure",
        "source_id": "local_generated_sts_default_residual_pressure",
        "split": "train",
        "category": family,
        "prompt": str(template["prompt"]),
        "entry_point": f"private_{family}_{template['name']}_{idx:04d}",
        "solution_expr": "",
        "solution_body": str(template["body"]),
        "tests": "",
        "tags": [
            "private_residual_curriculum",
            "sts_default",
            family,
            str(template["name"]),
            "no_public_tests",
            "no_public_solutions",
        ],
        "benchmark_evidence_level": "private_residual_generated_train_only",
        "public_benchmark": False,
        "license_spdx": "CC0-1.0",
        "candidate_expression_eligible": False,
        "decoder_contract": decoder_contract(
            category=family,
            return_shape=str(template["return_shape"]),
            arg_count=int(template["arg_count"]),
            required_constructs=list(template["required"]),
            skeleton_bias=list(template["bias"]),
        ),
        "provenance": {
            "policy": "project_theseus_sts_default_residual_pressure_v1",
            "residual_family": family,
            "aggregate_residual_count": source_count,
            "source_residual_packet": rel(residual_packet_path),
            "output_jsonl": rel(out_path),
            "public_benchmark_answers_used": False,
            "public_solutions_used": False,
            "public_tests_used": False,
            "public_prompts_used": False,
            "teacher_apply_used": False,
            "public_task_ids_hashed_only": True,
            "materialized_from_private_templates": True,
        },
    }


def build_rows(
    residual_counts: dict[str, int],
    rows_per_family: int,
    out_path: Path,
    residual_packet_path: Path,
) -> list[dict[str, Any]]:
    selected = [family for family in TARGET_FAMILIES if int(residual_counts.get(family, 0)) > 0]
    if not selected:
        selected = list(TARGET_FAMILIES)
    templates_by_family = {
        "algorithm_choice": algorithm_templates(),
        "local_code_generation_adapter_needed": adapter_templates(),
    }
    rows: list[dict[str, Any]] = []
    for family in selected:
        templates = templates_by_family[family]
        for idx in range(max(1, rows_per_family)):
            template = templates[idx % len(templates)]
            rows.append(
                build_row(
                    family=family,
                    idx=idx,
                    template=template,
                    source_count=int(residual_counts.get(family, 0)),
                    out_path=out_path,
                    residual_packet_path=residual_packet_path,
                )
            )
    return rows


def unsafe_public_flags(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("public_benchmark") is not False
        or bool(row.get("public_tests_used"))
        or bool(row.get("public_solutions_used"))
        or bool(row.get("public_benchmark_answers_used"))
        or bool(row.get("public_prompts_used"))
        or bool(get_path(row, ["provenance", "public_tests_used"], False))
        or bool(get_path(row, ["provenance", "public_solutions_used"], False))
        or bool(get_path(row, ["provenance", "public_benchmark_answers_used"], False))
        or bool(get_path(row, ["provenance", "public_prompts_used"], False))
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# STS Default Residual Pressure",
        "",
        f"- Trigger: `{report.get('trigger_state')}`",
        f"- Rows: `{summary.get('row_count', 0)}`",
        f"- Output: `{summary.get('jsonl_out', '')}`",
        f"- Unsafe public flags: `{summary.get('unsafe_public_flag_count', 0)}`",
        f"- Teacher apply used: `{summary.get('teacher_apply_used', False)}`",
        "",
        "## Family Counts",
    ]
    for family, count in sorted((summary.get("family_counts") or {}).items()):
        lines.append(f"- `{family}`: `{count}`")
    lines.extend(
        [
            "",
            "## Policy",
            "- Rows are generated from local private templates and aggregate residual family counts only.",
            "- Public benchmark prompts, tests, solutions, answers, and teacher apply are not used.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-packet", default=str(DEFAULT_RESIDUAL_PACKET.relative_to(ROOT)))
    parser.add_argument("--jsonl-out", default=rel(DEFAULT_JSONL))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--rows-per-family", type=int, default=64)
    args = parser.parse_args()

    residual_packet_path = resolve(args.residual_packet)
    jsonl_out = resolve(args.jsonl_out)
    out = resolve(args.out)
    markdown_out = resolve(args.markdown_out)
    residual_packet = read_json(residual_packet_path, {})
    residual_counts = dominant_residual_counts(residual_packet)
    rows = build_rows(residual_counts, max(1, int(args.rows_per_family)), jsonl_out, residual_packet_path)
    family_counts = Counter(str(row.get("category") or "") for row in rows)
    unsafe_flags = unsafe_public_flags(rows)
    write_jsonl(jsonl_out, rows)
    report = {
        "policy": "project_theseus_sts_default_residual_pressure_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if rows and unsafe_flags == 0 else "RED",
        "summary": {
            "row_count": len(rows),
            "family_counts": dict(sorted(family_counts.items())),
            "source_residual_counts": {family: int(residual_counts.get(family, 0)) for family in TARGET_FAMILIES},
            "residual_packet": rel(residual_packet_path),
            "jsonl_out": rel(jsonl_out),
            "unsafe_public_flag_count": unsafe_flags,
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_benchmark_answers_used": False,
            "public_prompts_used": False,
            "teacher_apply_used": False,
            "external_inference_calls": 0,
            "sts_default_training_pressure": True,
        },
        "gates": [
            {"name": "rows_written", "passed": bool(rows), "detail": len(rows)},
            {"name": "target_residuals_present", "passed": all(family in family_counts for family in TARGET_FAMILIES), "detail": dict(family_counts)},
            {"name": "public_content_not_used", "passed": unsafe_flags == 0, "detail": unsafe_flags},
            {"name": "teacher_apply_not_used", "passed": True, "detail": False},
        ],
    }
    write_json(out, report)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
