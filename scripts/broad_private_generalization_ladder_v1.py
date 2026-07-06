#!/usr/bin/env python3
"""Generate the Broad Private Generalization Ladder v1 curriculum.

This is a private-only broad transfer ladder. It deliberately does not read
public benchmark prompts, tests, solutions, score labels, or candidate code.
Rows are generated from local synthetic templates with executable private tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_rows import training_data_path  # noqa: E402
from code_residual_curriculum import verify_private_solution_rows  # noqa: E402


POLICY = "project_theseus_broad_private_generalization_ladder_v1"
TRAIN_DEFAULT = training_data_path(
    "high_transfer",
    "private_train",
    "broad_private_generalization_ladder_v1_code_lm_tasks.jsonl",
)
HELDOUT_DEFAULT = training_data_path(
    "high_transfer",
    "private_eval",
    "broad_private_generalization_ladder_v1_heldout_code_lm_tasks.jsonl",
)
FAMILIES = (
    "stdin_algorithmic",
    "graph_search",
    "dynamic_programming",
    "intervals",
    "state_machines",
    "parsing_encoding",
    "return_interface_fidelity",
    "data_structures",
    "numeric_edge_cases",
    "tool_style_transforms",
    "multi_step_contracts",
    "adversarial_metamorphic",
)


@dataclass(frozen=True)
class Template:
    family: str
    category: str
    entry: str
    prompt: str
    body: str
    tests: Callable[[str, int], str]
    return_shape: str
    type_family: str
    required_constructs: tuple[str, ...]
    tags: tuple[str, ...]
    visible_arg_count_hint: int = 1
    argument_roles: dict[str, str] | None = None
    semantic_family: str = ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=3000)
    parser.add_argument("--heldout-rows", type=int, default=1008)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--private-train-out", default=TRAIN_DEFAULT)
    parser.add_argument("--private-heldout-out", default=HELDOUT_DEFAULT)
    parser.add_argument("--out", default="reports/broad_private_generalization_ladder_v1.json")
    parser.add_argument("--markdown-out", default="reports/broad_private_generalization_ladder_v1.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    templates = template_bank()
    train_rows = build_rows(
        templates,
        row_count=max(240, int(args.train_rows)),
        split="train",
        seed=int(args.seed),
        id_offset=0,
    )
    heldout_rows = build_rows(
        templates,
        row_count=max(1000, int(args.heldout_rows)),
        split="heldout",
        seed=int(args.seed) + 100_000,
        id_offset=1_000_000,
    )
    train_path = resolve(args.private_train_out)
    heldout_path = resolve(args.private_heldout_out)
    write_jsonl(train_path, train_rows)
    write_jsonl(heldout_path, heldout_rows)

    train_check = verify_private_solution_rows(train_rows, max_failures=24)
    heldout_check = verify_private_solution_rows(heldout_rows, max_failures=24)
    train_family_counts = Counter(str(row.get("broad_private_family_v1")) for row in train_rows)
    heldout_family_counts = Counter(str(row.get("broad_private_family_v1")) for row in heldout_rows)
    train_category_counts = Counter(str(row.get("category")) for row in train_rows)
    heldout_category_counts = Counter(str(row.get("category")) for row in heldout_rows)
    leakage = public_leakage_scan(train_rows + heldout_rows)
    gates = [
        gate("private_train_rows_ge_2400", len(train_rows) >= 2400, len(train_rows)),
        gate("private_heldout_rows_ge_1000", len(heldout_rows) >= 1000, len(heldout_rows)),
        gate("required_family_count", set(train_family_counts) == set(FAMILIES), dict(train_family_counts)),
        gate("heldout_required_family_count", set(heldout_family_counts) == set(FAMILIES), dict(heldout_family_counts)),
        gate("category_diversity_ge_24", len(train_category_counts) >= 24 and len(heldout_category_counts) >= 24, {
            "train_categories": len(train_category_counts),
            "heldout_categories": len(heldout_category_counts),
        }),
        gate("private_train_solution_tests_pass", train_check["failure_count"] == 0, train_check),
        gate("private_heldout_solution_tests_pass", heldout_check["failure_count"] == 0, heldout_check),
        gate("public_data_leakage_zero", leakage["hit_count"] == 0, leakage),
        gate("external_inference_zero", True, 0),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "RED"
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": trigger_state,
        "purpose": "Broad private-only generalization ladder for unattended Theseus training/evaluation.",
        "inputs": {
            "seed": int(args.seed),
            "template_count": len(templates),
            "public_benchmark_inputs_read": False,
            "public_prompts_used": False,
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_score_labels_used": False,
        },
        "outputs": {
            "private_train_jsonl": rel(train_path),
            "private_heldout_jsonl": rel(heldout_path),
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
        },
        "summary": {
            "private_train_row_count": len(train_rows),
            "private_heldout_row_count": len(heldout_rows),
            "family_train_row_counts": dict(sorted(train_family_counts.items())),
            "family_heldout_row_counts": dict(sorted(heldout_family_counts.items())),
            "category_train_count": len(train_category_counts),
            "category_heldout_count": len(heldout_category_counts),
            "private_train_solution_failures": train_check["failure_count"],
            "private_heldout_solution_failures": heldout_check["failure_count"],
            "public_data_leakage_hit_count": leakage["hit_count"],
            "external_inference_calls": 0,
            "score_semantics": "private synthetic broad generalization pressure only; not public calibration",
        },
        "families": family_reports(train_rows, heldout_rows),
        "gates": gates,
        "next_actions": [
            "run the unattended ladder runner with --execute and either --train or --checkpoint-in",
            "fan out candidates against the heldout split with STS-on and same-seed STS-off control",
            "score broad private heldout and write the broad private generalization gate",
            "keep public calibration locked throughout",
        ],
        "external_inference_calls": 0,
    }


def build_rows(
    templates: list[Template],
    *,
    row_count: int,
    split: str,
    seed: int,
    id_offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        template = templates[(index + seed) % len(templates)]
        variant = seed + index * 17
        task_index = id_offset + index
        rows.append(row_from_template(template, split=split, task_index=task_index, variant=variant))
    return rows


def row_from_template(template: Template, *, split: str, task_index: int, variant: int) -> dict[str, Any]:
    entry = f"{template.entry}_{task_index:07d}"
    tags = sorted(
        {
            "broad_private_generalization_ladder_v1",
            split,
            template.family,
            template.category,
            *template.tags,
        }
    )
    return {
        "task_id": f"broad_private_generalization_ladder_v1_{template.family}_{task_index:07d}",
        "source_task_id": f"broad_private_generalization_ladder_v1_{split}_{variant:07d}",
        "card_id": "broad_private_generalization_ladder_v1",
        "source_id": "local_generated_broad_private_generalization_ladder_v1",
        "split": "train" if split == "train" else "eval",
        "category": template.category,
        "prompt": template.prompt,
        "entry_point": entry,
        "solution_expr": "",
        "solution_body": template.body,
        "tests": normalize_test_source(template.tests(entry, variant)),
        "tags": tags,
        "broad_private_family_v1": template.family,
        "targeted_private_residual_family_v3": template.family,
        "residual_concept": template.semantic_family or template.category,
        "concept_residual_label": template.category,
        "metamorphic_properties": metamorphic_properties(template),
        "decoder_contract": decoder_contract(template),
        "benchmark_evidence_level": "broad_private_generalization_ladder_v1_generated_only",
        "public_benchmark": False,
        "public_benchmark_solutions_included": False,
        "public_tests_included": False,
        "public_prompts_included": False,
        "public_score_labels_included": False,
        "license_spdx": "CC0-1.0",
        "candidate_expression_eligible": False,
        "provenance": {
            "policy": POLICY,
            "family": template.family,
            "category": template.category,
            "variant": variant,
            "public_benchmark_answers_used": False,
            "public_tests_used": False,
            "public_prompts_used": False,
            "public_score_labels_used": False,
            "semantics": "private synthetic broad generalization pressure only",
        },
    }


def decoder_contract(template: Template) -> dict[str, Any]:
    return {
        "policy": "project_theseus_decoder_contract_v1_broad_private_generalization",
        "return_shape": template.return_shape,
        "type_family": template.type_family,
        "semantic_family": template.semantic_family or template.category,
        "visible_arg_count_hint": template.visible_arg_count_hint,
        "required_constructs": list(template.required_constructs),
        "residual_label_hint": template.category,
        "full_body_required": True,
        "guardrail_only": False,
        "feedback_weight": 1.35,
        "score_semantics": "private broad generalization pressure only",
        "argument_roles": template.argument_roles or {"data": "primary_input"},
        "return_contract": {
            "shape": template.return_shape,
            "empty_or_invalid_behavior": "covered_by_private_assertions",
            "must_preserve_container_shape": template.return_shape in {"list", "dict", "tuple"},
        },
        "generation_plan": {
            "policy": "signature -> semantic_family -> private_tests -> return_contract -> branch_loop_state -> body",
            "skeleton_bias": list(template.required_constructs),
            "repair_strategy": "prefer reusable semantic-family bodies over benchmark-specific adapters",
            "public_tests_used": False,
            "public_solutions_used": False,
        },
    }


def normalize_test_source(source: str) -> str:
    """Convert escaped statement separators while preserving string literal newlines."""
    text = source.replace("\\nassert ", "\nassert ")
    if text.endswith("\\n"):
        text = text[:-2] + "\n"
    return text


def metamorphic_properties(template: Template) -> list[str]:
    by_family = {
        "stdin_algorithmic": ["stdin_whitespace_tolerance", "newline_output_contract"],
        "graph_search": ["node_relabeling_invariance", "component_boundary_cases"],
        "dynamic_programming": ["prefix_optimal_substructure", "empty_input_boundary"],
        "intervals": ["sort_invariance", "overlap_boundary"],
        "state_machines": ["state_reset_boundary", "stream_order_sensitivity"],
        "parsing_encoding": ["roundtrip_or_idempotence", "malformed_input_guard"],
        "return_interface_fidelity": ["exact_shape_contract", "empty_return_contract"],
        "data_structures": ["duplicate_handling", "stable_order_or_priority"],
        "numeric_edge_cases": ["negative_and_zero_boundary", "tolerance_or_modulo_guard"],
        "tool_style_transforms": ["field_projection", "stable_grouping"],
        "multi_step_contracts": ["normalize_filter_group_sort", "pipeline_order_matters"],
        "adversarial_metamorphic": ["malformed_input_guard", "idempotence_or_duplicate_replay"],
    }
    return by_family.get(template.family, [])


def template_bank() -> list[Template]:
    return [
        Template("stdin_algorithmic", "bpg_stdin_pair_sums", "bpg_stdin_pair_sums", "Read lines containing integer pairs and return one sum per line.", pair_sum_body(), tests_pair_sums, "str", "stdin_numeric_line_parser", ("loop", "branch", "locals", "stdin_parse", "string_join_return"), ("stdin", "competitive_programming")),
        Template("stdin_algorithmic", "bpg_stdin_prefix_queries", "bpg_stdin_prefix_queries", "Parse n/q prefix-sum range queries from a whitespace stdin string.", prefix_query_body(), tests_prefix_queries, "str", "algorithmic_planning", ("loop", "branch", "locals", "stdin_parse", "algorithmic_planning"), ("stdin", "prefix_sum")),
        Template("graph_search", "bpg_graph_components", "bpg_graph_components", "Return the number of connected components in an undirected graph edge list.", graph_components_body(), tests_graph_components, "number", "graph_search_algorithm", ("loop", "branch", "locals", "graph", "algorithmic_planning"), ("graph", "dfs"), 2, {"data": "node_count", "other": "edge_list"}),
        Template("graph_search", "bpg_shortest_hops", "bpg_shortest_hops", "Return shortest unweighted hop count between two nodes, or -1 when unreachable.", shortest_hops_body(), tests_shortest_hops, "number", "graph_search_algorithm", ("loop", "branch", "locals", "graph", "algorithmic_planning"), ("graph", "bfs"), 4, {"data": "node_count", "other": "edge_list", "start": "source", "goal": "target"}),
        Template("dynamic_programming", "bpg_max_non_adjacent_sum", "bpg_max_non_adjacent_sum", "Return the maximum sum of non-adjacent numeric values.", max_non_adjacent_body(), tests_max_non_adjacent, "number", "dynamic_programming", ("loop", "branch", "locals", "algorithmic_planning"), ("dynamic_programming", "state_update")),
        Template("dynamic_programming", "bpg_lcs_length", "bpg_lcs_length", "Return longest common subsequence length for two strings.", lcs_length_body(), tests_lcs_length, "number", "dynamic_programming", ("loop", "branch", "locals", "algorithmic_planning", "index_or_string_ops"), ("dynamic_programming", "string")),
        Template("intervals", "bpg_merge_intervals", "bpg_merge_intervals", "Merge half-open intervals and return sorted non-overlapping intervals.", merge_intervals_body(), tests_merge_intervals, "list", "grouped_interval_algorithm", ("loop", "branch", "locals", "algorithmic_planning", "type_and_return_shape"), ("intervals", "merge")),
        Template("intervals", "bpg_interval_coverage", "bpg_interval_coverage", "Return total covered half-open interval length after merging overlaps.", interval_coverage_body(), tests_interval_coverage, "number", "grouped_interval_algorithm", ("loop", "branch", "locals", "algorithmic_planning"), ("intervals", "coverage")),
        Template("state_machines", "bpg_longest_even_run", "bpg_longest_even_run", "Return length of the longest contiguous run of even integers.", longest_even_run_body(), tests_longest_even_run, "number", "state_machine", ("loop", "branch", "locals", "algorithmic_planning"), ("state_machine", "run_length")),
        Template("state_machines", "bpg_parse_signed_ints", "bpg_parse_signed_ints", "Extract signed integers from noisy text without using regex.", parse_signed_ints_body(), tests_parse_signed_ints, "list", "state_machine_parser", ("loop", "branch", "locals", "parsing", "type_and_return_shape"), ("state_machine", "parser")),
        Template("parsing_encoding", "bpg_rle_encode", "bpg_rle_encode", "Run-length encode consecutive equal values as (value, count) pairs.", rle_encode_body(), tests_rle_encode, "list", "collection_logic", ("loop", "branch", "locals", "collection_ops", "type_and_return_shape"), ("encoding", "roundtrip")),
        Template("parsing_encoding", "bpg_parse_query_string", "bpg_parse_query_string", "Parse a query string into a dict of key -> list of values.", parse_query_body(), tests_parse_query, "dict", "structured_parsing", ("loop", "branch", "locals", "parsing", "type_and_return_shape"), ("parsing", "dict")),
        Template("return_interface_fidelity", "bpg_numeric_stats_tuple", "bpg_numeric_stats_tuple", "Return (min, max, count) for numeric non-bool values.", numeric_stats_tuple_body(), tests_numeric_stats_tuple, "tuple", "heterogeneous_numeric_contract", ("loop", "branch", "locals", "type_and_return_shape"), ("return_shape", "tuple")),
        Template("return_interface_fidelity", "bpg_threshold_labels", "bpg_threshold_labels", "Return labels from records whose numeric score meets a threshold.", threshold_labels_body(), tests_threshold_labels, "list", "collection_logic", ("loop", "branch", "locals", "two_arg_interface", "type_and_return_shape"), ("return_shape", "two_arg"), 2, {"data": "records", "other": "threshold"}),
        Template("data_structures", "bpg_top_k_frequent", "bpg_top_k_frequent", "Return top-k frequent values ordered by frequency descending then value ascending.", top_k_frequent_body(), tests_top_k_frequent, "list", "collection_logic", ("loop", "branch", "locals", "collection_ops", "type_and_return_shape"), ("data_structures", "frequency"), 2, {"data": "values", "other": "k"}),
        Template("data_structures", "bpg_stable_dedup", "bpg_stable_dedup", "Return first-seen unique normalized tokens while preserving encounter order.", stable_dedup_body(), tests_stable_dedup, "list", "collection_transform", ("loop", "branch", "locals", "collection_ops", "type_and_return_shape"), ("data_structures", "dedup")),
        Template("numeric_edge_cases", "bpg_clamp_round", "bpg_clamp_round", "Clamp numeric values into a range and round to a fixed number of digits.", clamp_round_body(), tests_clamp_round, "list", "numeric_transform", ("loop", "branch", "locals", "numeric_ops", "type_and_return_shape"), ("numeric", "edge_cases"), 2, {"data": "values", "other": "(lo, hi, digits)"}),
        Template("numeric_edge_cases", "bpg_gcd_positive", "bpg_gcd_positive", "Return gcd of absolute positive integer values, ignoring bools and non-integers.", gcd_positive_body(), tests_gcd_positive, "number", "numeric_algorithm", ("loop", "branch", "locals", "numeric_ops"), ("numeric", "gcd")),
        Template("tool_style_transforms", "bpg_group_records", "bpg_group_records", "Group record ids by a named field, skipping malformed records.", group_records_body(), tests_group_records, "dict", "record_transform", ("loop", "branch", "locals", "collection_ops", "type_and_return_shape"), ("tool_transform", "records"), 2, {"data": "records", "other": "field"}),
        Template("tool_style_transforms", "bpg_project_table", "bpg_project_table", "Project rows to selected columns with default None for missing fields.", project_table_body(), tests_project_table, "list", "record_transform", ("loop", "branch", "locals", "collection_ops", "type_and_return_shape"), ("tool_transform", "projection"), 2, {"data": "rows", "other": "columns"}),
        Template("multi_step_contracts", "bpg_normalize_filter_sort", "bpg_normalize_filter_sort", "Normalize text tokens, filter stopwords/short tokens, return sorted unique tokens.", normalize_filter_sort_body(), tests_normalize_filter_sort, "list", "multi_step_pipeline", ("loop", "branch", "locals", "collection_ops", "index_or_string_ops", "type_and_return_shape"), ("multi_step", "normalize_filter_sort"), 2, {"data": "tokens", "other": "stopwords"}),
        Template("multi_step_contracts", "bpg_windowed_deltas", "bpg_windowed_deltas", "Return adjacent deltas for numeric values after clipping to a range.", windowed_deltas_body(), tests_windowed_deltas, "list", "multi_step_numeric_pipeline", ("loop", "branch", "locals", "numeric_ops", "type_and_return_shape"), ("multi_step", "numeric_pipeline"), 2, {"data": "values", "other": "(lo, hi)"}),
        Template("adversarial_metamorphic", "bpg_safe_head_default", "bpg_safe_head_default", "Return the first sequence item or a default for empty/non-sequence input.", safe_head_body(), tests_safe_head, "unknown", "interface_fidelity", ("branch", "locals", "two_arg_interface"), ("adversarial", "edge_cases"), 2, {"data": "sequence", "other": "default"}),
        Template("adversarial_metamorphic", "bpg_balanced_parens", "bpg_balanced_parens", "Return True when bracket characters are balanced, ignoring other text.", balanced_parens_body(), tests_balanced_parens, "bool", "state_machine", ("loop", "branch", "locals", "algorithmic_planning"), ("adversarial", "state_machine")),
    ]


def pair_sum_body() -> str:
    return "out = []\nfor line in str(data).splitlines():\n    parts = line.split()\n    if len(parts) < 2:\n        continue\n    try:\n        out.append(str(int(parts[0]) + int(parts[1])))\n    except Exception:\n        continue\nreturn '\\n'.join(out)"


def prefix_query_body() -> str:
    return "try:\n    tokens = [int(x) for x in str(data).split()]\nexcept Exception:\n    return ''\nif len(tokens) < 2:\n    return ''\nn, q = tokens[0], tokens[1]\nvalues = tokens[2:2+n]\nprefix = [0]\nfor value in values:\n    prefix.append(prefix[-1] + value)\nout = []\npos = 2 + n\nfor _ in range(q):\n    if pos + 1 >= len(tokens):\n        break\n    left, right = tokens[pos], tokens[pos + 1]\n    pos += 2\n    left = max(1, left)\n    right = min(n, right)\n    out.append(str(prefix[right] - prefix[left - 1] if left <= right else 0))\nreturn '\\n'.join(out)"


def graph_components_body() -> str:
    return "graph = [[] for _ in range(max(0, int(data)))]\nfor edge in other:\n    if not isinstance(edge, (list, tuple)) or len(edge) < 2:\n        continue\n    try:\n        a, b = int(edge[0]), int(edge[1])\n    except Exception:\n        continue\n    if 0 <= a < len(graph) and 0 <= b < len(graph):\n        graph[a].append(b)\n        graph[b].append(a)\nseen = set()\ncomponents = 0\nfor start in range(len(graph)):\n    if start in seen:\n        continue\n    components += 1\n    stack = [start]\n    seen.add(start)\n    while stack:\n        node = stack.pop()\n        for nxt in graph[node]:\n            if nxt not in seen:\n                seen.add(nxt)\n                stack.append(nxt)\nreturn components"


def shortest_hops_body() -> str:
    return "from collections import deque\nstart = extra[0] if len(extra) > 0 else 0\ngoal = extra[1] if len(extra) > 1 else 0\ngraph = [[] for _ in range(max(0, int(data)))]\nfor edge in other:\n    if not isinstance(edge, (list, tuple)) or len(edge) < 2:\n        continue\n    try:\n        a, b = int(edge[0]), int(edge[1])\n    except Exception:\n        continue\n    if 0 <= a < len(graph) and 0 <= b < len(graph):\n        graph[a].append(b)\n        graph[b].append(a)\nif start < 0 or goal < 0 or start >= len(graph) or goal >= len(graph):\n    return -1\nqueue = deque([(start, 0)])\nseen = {start}\nwhile queue:\n    node, dist = queue.popleft()\n    if node == goal:\n        return dist\n    for nxt in graph[node]:\n        if nxt not in seen:\n            seen.add(nxt)\n            queue.append((nxt, dist + 1))\nreturn -1"


def max_non_adjacent_body() -> str:
    return "take = 0\nskip = 0\nfor value in data:\n    value = max(0, int(value))\n    take, skip = skip + value, max(skip, take)\nreturn max(take, skip)"


def lcs_length_body() -> str:
    return "a = str(data)\nb = str(other)\nprev = [0] * (len(b) + 1)\nfor ch_a in a:\n    cur = [0]\n    for j, ch_b in enumerate(b, 1):\n        if ch_a == ch_b:\n            cur.append(prev[j - 1] + 1)\n        else:\n            cur.append(max(prev[j], cur[-1]))\n    prev = cur\nreturn prev[-1]"


def merge_intervals_body() -> str:
    return "intervals = []\nfor item in data:\n    if isinstance(item, (list, tuple)) and len(item) >= 2:\n        a, b = item[0], item[1]\n        if b > a:\n            intervals.append((a, b))\nmerged = []\nfor a, b in sorted(intervals):\n    if not merged or a > merged[-1][1]:\n        merged.append([a, b])\n    else:\n        merged[-1][1] = max(merged[-1][1], b)\nreturn [tuple(item) for item in merged]"


def interval_coverage_body() -> str:
    return "intervals = []\nfor item in data:\n    if isinstance(item, (list, tuple)) and len(item) >= 2 and item[1] > item[0]:\n        intervals.append((item[0], item[1]))\nmerged = []\nfor a, b in sorted(intervals):\n    if not merged or a > merged[-1][1]:\n        merged.append([a, b])\n    else:\n        merged[-1][1] = max(merged[-1][1], b)\nreturn sum(b - a for a, b in merged)"


def longest_even_run_body() -> str:
    return "best = 0\ncurrent = 0\nfor value in data:\n    if int(value) % 2 == 0:\n        current += 1\n        best = max(best, current)\n    else:\n        current = 0\nreturn best"


def parse_signed_ints_body() -> str:
    return "out = []\nnum = ''\nsign = ''\nfor ch in str(data) + ' ':\n    if ch in '+-' and not num:\n        sign = ch\n    elif ch.isdigit():\n        num += ch\n    else:\n        if num:\n            out.append(int((sign or '') + num))\n        num = ''\n        sign = ''\nreturn out"


def rle_encode_body() -> str:
    return "out = []\nfor item in data:\n    if out and out[-1][0] == item:\n        out[-1] = (item, out[-1][1] + 1)\n    else:\n        out.append((item, 1))\nreturn out"


def parse_query_body() -> str:
    return "out = {}\ntext = str(data).lstrip('?')\nfor part in text.split('&'):\n    if not part:\n        continue\n    if '=' in part:\n        key, value = part.split('=', 1)\n    else:\n        key, value = part, ''\n    out.setdefault(key, []).append(value)\nreturn out"


def numeric_stats_tuple_body() -> str:
    return "values = [item for item in data if isinstance(item, (int, float)) and not isinstance(item, bool)]\nif not values:\n    return (None, None, 0)\nreturn (min(values), max(values), len(values))"


def threshold_labels_body() -> str:
    return "out = []\nfor record in data:\n    if not isinstance(record, dict):\n        continue\n    try:\n        score = float(record.get('score', 0))\n    except Exception:\n        score = 0.0\n    if score >= other and record.get('label') is not None:\n        out.append(str(record.get('label')))\nreturn out"


def top_k_frequent_body() -> str:
    return "counts = {}\nfor item in data:\n    counts[item] = counts.get(item, 0) + 1\nitems = sorted(counts, key=lambda key: (-counts[key], key))\nreturn items[:max(0, int(other))]"


def stable_dedup_body() -> str:
    return "seen = set()\nout = []\nfor item in data:\n    text = str(item).strip().casefold()\n    if not text or text in seen:\n        continue\n    seen.add(text)\n    out.append(text)\nreturn out"


def clamp_round_body() -> str:
    return "lo, hi, digits = other\nout = []\nfor value in data:\n    try:\n        number = float(value)\n    except Exception:\n        continue\n    number = min(max(number, lo), hi)\n    out.append(round(number, int(digits)))\nreturn out"


def gcd_positive_body() -> str:
    return "import math\nanswer = 0\nfor value in data:\n    if isinstance(value, bool) or not isinstance(value, int):\n        continue\n    value = abs(value)\n    if value:\n        answer = math.gcd(answer, value)\nreturn answer"


def group_records_body() -> str:
    return "out = {}\nfor record in data:\n    if not isinstance(record, dict) or 'id' not in record:\n        continue\n    key = record.get(other)\n    if key is None:\n        continue\n    out.setdefault(str(key), []).append(record['id'])\nreturn out"


def project_table_body() -> str:
    return "out = []\nfor row in data:\n    if not isinstance(row, dict):\n        continue\n    out.append({col: row.get(col) for col in other})\nreturn out"


def normalize_filter_sort_body() -> str:
    return "stop = {str(item).casefold() for item in other}\nout = set()\nfor item in data:\n    text = str(item).strip().casefold()\n    if len(text) < 2 or text in stop:\n        continue\n    out.add(text)\nreturn sorted(out)"


def windowed_deltas_body() -> str:
    return "lo, hi = other\nvalues = []\nfor value in data:\n    try:\n        values.append(min(max(float(value), lo), hi))\n    except Exception:\n        continue\nreturn [values[i + 1] - values[i] for i in range(len(values) - 1)]"


def safe_head_body() -> str:
    return "if isinstance(data, (list, tuple)) and data:\n    return data[0]\nreturn other"


def balanced_parens_body() -> str:
    return "pairs = {')': '(', ']': '[', '}': '{'}\nstack = []\nfor ch in str(data):\n    if ch in '([{':\n        stack.append(ch)\n    elif ch in pairs:\n        if not stack or stack[-1] != pairs[ch]:\n            return False\n        stack.pop()\nreturn not stack"


def tests_pair_sums(entry: str, variant: int) -> str:
    a = variant % 31 - 15
    b = variant % 17
    return f"assert {entry}('{a} {b}\\n1 2\\nbad\\n') == '{a + b}\\n3'\\nassert {entry}('') == ''\\n"


def tests_prefix_queries(entry: str, variant: int) -> str:
    base = [variant % 5 + 1, 2, 3, 4]
    total = sum(base)
    return f"assert {entry}('4 2\\n{' '.join(map(str, base))}\\n1 4\\n2 3\\n') == '{total}\\n{base[1] + base[2]}'\\nassert {entry}('bad') == ''\\n"


def tests_graph_components(entry: str, variant: int) -> str:
    return f"assert {entry}(5, [(0, 1), (3, 4)]) == 3\\nassert {entry}(1, []) == 1\\nassert {entry}(3, [(0, 9), ('x', 2)]) == 3\\n"


def tests_shortest_hops(entry: str, variant: int) -> str:
    return f"assert {entry}(5, [(0, 1), (1, 2), (3, 4)], 0, 2) == 2\\nassert {entry}(5, [(0, 1)], 0, 4) == -1\\nassert {entry}(2, [], 4, 1) == -1\\n"


def tests_max_non_adjacent(entry: str, variant: int) -> str:
    return f"assert {entry}([2, 7, 9, 3, 1]) == 12\\nassert {entry}([-5, 4, 6]) == 6\\nassert {entry}([]) == 0\\n"


def tests_lcs_length(entry: str, variant: int) -> str:
    return f"assert {entry}('abcde', 'ace') == 3\\nassert {entry}('', 'abc') == 0\\nassert {entry}('abc', 'def') == 0\\n"


def tests_merge_intervals(entry: str, variant: int) -> str:
    return f"assert {entry}([(5, 7), (1, 3), (2, 5)]) == [(1, 7)]\\nassert {entry}([(1, 1), (2, 4)]) == [(2, 4)]\\nassert {entry}([]) == []\\n"


def tests_interval_coverage(entry: str, variant: int) -> str:
    return f"assert {entry}([(1, 3), (2, 6), (9, 10)]) == 6\\nassert {entry}([(5, 5), (7, 6)]) == 0\\nassert {entry}([]) == 0\\n"


def tests_longest_even_run(entry: str, variant: int) -> str:
    return f"assert {entry}([2, 4, 1, 6, 8, 10]) == 3\\nassert {entry}([1, 3]) == 0\\nassert {entry}([]) == 0\\n"


def tests_parse_signed_ints(entry: str, variant: int) -> str:
    return f"assert {entry}('a-12 b +7 003x') == [-12, 7, 3]\\nassert {entry}('none') == []\\nassert {entry}('- 5') == [5]\\n"


def tests_rle_encode(entry: str, variant: int) -> str:
    return f"assert {entry}(['a', 'a', 'b']) == [('a', 2), ('b', 1)]\\nassert {entry}([]) == []\\nassert {entry}(['x', 'y', 'x']) == [('x', 1), ('y', 1), ('x', 1)]\\n"


def tests_parse_query(entry: str, variant: int) -> str:
    return f"assert {entry}('?a=1&b=2&a=3&empty=') == {{'a': ['1', '3'], 'b': ['2'], 'empty': ['']}}\\nassert {entry}('flag&x=') == {{'flag': [''], 'x': ['']}}\\n"


def tests_numeric_stats_tuple(entry: str, variant: int) -> str:
    return f"assert {entry}([3, 1, True, 'x']) == (1, 3, 2)\\nassert {entry}([]) == (None, None, 0)\\nassert isinstance({entry}([1, 2]), tuple)\\n"


def tests_threshold_labels(entry: str, variant: int) -> str:
    return f"assert {entry}([{{'label': 'a', 'score': 2}}, {{'label': 'b', 'score': '3'}}], 2.5) == ['b']\\nassert {entry}([], 1) == []\\nassert {entry}([{{'score': 9}}, 1, {{'label': 'x', 'score': 'bad'}}], 0) == ['x']\\n"


def tests_top_k_frequent(entry: str, variant: int) -> str:
    return f"assert {entry}(['b', 'a', 'b', 'c', 'a', 'b'], 2) == ['b', 'a']\\nassert {entry}([], 3) == []\\n"


def tests_stable_dedup(entry: str, variant: int) -> str:
    return f"assert {entry}([' A ', 'a', 'B', '', 'b']) == ['a', 'b']\\nassert {entry}(['x', 'X', 'x ']) == ['x']\\n"


def tests_clamp_round(entry: str, variant: int) -> str:
    return f"assert {entry}([-2, 0.1234, 9], (0, 1, 2)) == [0, 0.12, 1]\\nassert {entry}(['bad', 2], (0, 5, 0)) == [2.0]\\n"


def tests_gcd_positive(entry: str, variant: int) -> str:
    return f"assert {entry}([12, -18, True, 'x']) == 6\\nassert {entry}([]) == 0\\nassert {entry}([7]) == 7\\n"


def tests_group_records(entry: str, variant: int) -> str:
    return f"assert {entry}([{{'id': 1, 'kind': 'a'}}, {{'id': 2, 'kind': 'a'}}, {{'id': 3, 'kind': 'b'}}, {{'kind': 'x'}}], 'kind') == {{'a': [1, 2], 'b': [3]}}\\nassert {entry}([], 'kind') == {{}}\\n"


def tests_project_table(entry: str, variant: int) -> str:
    return f"assert {entry}([{{'a': 1, 'b': 2}}, {{'a': 3}}], ['a', 'b']) == [{{'a': 1, 'b': 2}}, {{'a': 3, 'b': None}}]\\nassert {entry}([1], ['a']) == []\\n"


def tests_normalize_filter_sort(entry: str, variant: int) -> str:
    return f"assert {entry}([' A ', 'bb', 'a', 'CC', 'bb'], ['cc']) == ['bb']\\nassert {entry}([], []) == []\\n"


def tests_windowed_deltas(entry: str, variant: int) -> str:
    return f"assert {entry}([-5, 0, 10], (0, 5)) == [0.0, 5.0]\\nassert {entry}(['bad', 1], (0, 2)) == []\\n"


def tests_safe_head(entry: str, variant: int) -> str:
    return f"assert {entry}([1, 2], 9) == 1\\nassert {entry}([], 'x') == 'x'\\nassert {entry}(None, 0) == 0\\nassert {entry}(('a',), 'z') == 'a'\\n"


def tests_balanced_parens(entry: str, variant: int) -> str:
    return f"assert {entry}('a(b[c]{{d}})') is True\\nassert {entry}('([)]') is False\\nassert {entry}('text') is True\\n"


def family_reports(train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for family in FAMILIES:
        train = [row for row in train_rows if row.get("broad_private_family_v1") == family]
        heldout = [row for row in heldout_rows if row.get("broad_private_family_v1") == family]
        out.append(
            {
                "family": family,
                "train_rows": len(train),
                "heldout_rows": len(heldout),
                "categories": sorted({str(row.get("category")) for row in train}),
                "decoder_contract_rows": sum(1 for row in train if isinstance(row.get("decoder_contract"), dict)),
            }
        )
    return out


def public_leakage_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    needles = [
        "humaneval",
        "mbpp",
        "evalplus",
        "bigcodebench",
        "livecodebench",
        "canonical_solution",
        "public_test",
    ]
    hits = []
    for row in rows:
        text = "\n".join(leakage_strings(row)).lower()
        for needle in needles:
            if needle in text:
                hits.append({"task_id": row.get("task_id"), "needle": needle})
                break
        if len(hits) >= 20:
            break
    return {"hit_count": len(hits), "sample_hits": hits}


def leakage_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, str):
        return [value]
    return []


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Broad Private Generalization Ladder V1",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Private train rows: {summary.get('private_train_row_count')}",
        f"- Private heldout rows: {summary.get('private_heldout_row_count')}",
        f"- Train categories: {summary.get('category_train_count')}",
        f"- Heldout categories: {summary.get('category_heldout_count')}",
        f"- Train solution failures: {summary.get('private_train_solution_failures')}",
        f"- Heldout solution failures: {summary.get('private_heldout_solution_failures')}",
        f"- Public-data leakage hits: {summary.get('public_data_leakage_hit_count')}",
        "",
        "## Families",
    ]
    for row in report.get("families", []):
        lines.append(
            f"- `{row.get('family')}`: train {row.get('train_rows')}, heldout {row.get('heldout_rows')}, categories {len(row.get('categories') or [])}"
        )
    lines.extend(["", "Public benchmark prompts, tests, solutions, and score labels are not used."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
