#!/usr/bin/env python3
"""Build a deterministic wide public-code calibration slice.

The output manifest pins public benchmark task IDs for calibration only. It
does not emit prompts, tests, reference solutions, or candidate code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_code_benchmark_graduation as real_code  # noqa: E402
from public_code_case_manifest import (  # noqa: E402
    PUBLIC_CASE_MANIFEST_CONTENT_POLICY,
    PUBLIC_CASE_MANIFEST_POLICY,
    load_case_manifest,
)


DEFAULT_CARDS = "source_mbpp,source_evalplus,source_bigcodebench,source_human_eval,source_livecodebench"
FEATURE_KEYS = [
    "interface_shape",
    "return_shape",
    "parsing_burden",
    "dependency_shape",
    "algorithm_family",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default=DEFAULT_CARDS)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--cases-per-card", type=int, default=32)
    parser.add_argument("--pool-cases-per-card", type=int, default=4096)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="Existing public selector manifest(s) to exclude so a new slice is disjoint by card/task id.",
    )
    parser.add_argument("--out", default="reports/public_wide_slice_manifest_seed23_5x32.jsonl")
    parser.add_argument("--report-out", default="reports/public_wide_slice_selector_seed23_5x32.json")
    parser.add_argument("--markdown-out", default="reports/public_wide_slice_selector_seed23_5x32.md")
    args = parser.parse_args()

    cards = real_code.expand_requested_cards([card.strip() for card in args.cards.split(",") if card.strip()])
    cases_per_card = max(1, int(args.cases_per_card))
    pool_cases = max(cases_per_card, int(args.pool_cases_per_card), cases_per_card * 8)
    manifest_rows: list[dict[str, Any]] = []
    card_reports: list[dict[str, Any]] = []
    excluded_task_keys = load_excluded_task_keys(args.exclude_manifest)

    for card_id in cards:
        card_report, selected = select_card_slice(
            card_id,
            seed=int(args.seed),
            cases_per_card=cases_per_card,
            pool_cases=pool_cases,
            excluded_task_keys=excluded_task_keys,
        )
        card_reports.append(card_report)
        manifest_rows.extend(selected)

    ready = all(row.get("ready_for_wide_public_calibration") for row in card_reports)
    report = {
        "policy": "project_theseus_wide_public_code_slice_selector_v1",
        "manifest_policy": PUBLIC_CASE_MANIFEST_POLICY,
        "created_utc": real_code.now(),
        "trigger_state": "GREEN" if ready else "YELLOW",
        "ready_for_wide_public_calibration": ready,
        "seed": int(args.seed),
        "cards": cards,
        "cases_per_card": cases_per_card,
        "required_total_cases": len(cards) * cases_per_card,
        "selected_total_cases": len(manifest_rows),
        "content_policy": PUBLIC_CASE_MANIFEST_CONTENT_POLICY,
        "public_training_rule": "public task IDs are calibration selectors only; private rows must not contain public prompts, tests, solutions, or score labels",
        "excluded_manifest_count": len(normalize_manifest_paths(args.exclude_manifest)),
        "excluded_task_key_count": len(excluded_task_keys),
        "cards_report": card_reports,
        "feature_distribution": summarize_manifest_features(manifest_rows),
        "blockers": [blocker for row in card_reports for blocker in row.get("blockers", [])],
        "artifacts": {
            "manifest": real_code.rel(real_code.resolve(args.out)),
            "report": real_code.rel(real_code.resolve(args.report_out)),
            "markdown": real_code.rel(real_code.resolve(args.markdown_out)),
        },
        "external_inference_calls": 0,
    }

    real_code.write_jsonl(real_code.resolve(args.out), manifest_rows)
    real_code.write_json(real_code.resolve(args.report_out), report)
    real_code.resolve(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
    real_code.resolve(args.markdown_out).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 1


def select_card_slice(
    card_id: str,
    *,
    seed: int,
    cases_per_card: int,
    pool_cases: int,
    excluded_task_keys: set[tuple[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    card = real_code.read_json(ROOT / "benchmarks" / "cards" / f"{card_id}.json", {})
    source_id = str(card.get("source_id") or card_id.replace("source_", ""))
    source_path = real_code.resolve_source_path(card)
    tasks: list[dict[str, Any]] = []
    evidence_level = "source_missing"
    semantics = "blocked_not_scored"
    if source_path.exists():
        tasks, evidence_level, semantics = real_code.load_cases(card_id, source_id, source_path, seed, pool_cases)
    else:
        fallback_tasks, fallback_evidence, fallback_semantics = real_code.load_cases(
            card_id,
            source_id,
            source_path,
            seed,
            min(pool_cases, cases_per_card),
        )
        tasks = fallback_tasks
        evidence_level = fallback_evidence
        semantics = fallback_semantics
    unfiltered_task_count = len(tasks)
    if excluded_task_keys:
        tasks = [
            task
            for task in tasks
            if (card_id, str(task.get("task_id") or "")) not in excluded_task_keys
        ]

    real_public = evidence_level == "public_benchmark_task_regression" and source_path.exists()
    blockers: list[dict[str, Any]] = []
    if not source_path.exists():
        blockers.append(
            {
                "card_id": card_id,
                "blocker": "source_not_staged",
                "source_path": real_code.rel_or_abs(source_path),
                "detail": "Stage the real benchmark payload locally before claiming this card in a wide public calibration.",
            }
        )
    if evidence_level != "public_benchmark_task_regression":
        blockers.append(
            {
                "card_id": card_id,
                "blocker": "not_real_public_benchmark_task_regression",
                "benchmark_evidence_level": evidence_level,
                "detail": "Loader or metadata regression tasks cannot count toward the promotion-grade public slice.",
            }
        )
    if len(tasks) < cases_per_card:
        blockers.append(
            {
                "card_id": card_id,
                "blocker": "insufficient_local_real_public_task_cases",
                "available_task_count": len(tasks),
                "required_task_count": cases_per_card,
            }
        )

    selected_tasks = select_diverse_tasks(tasks, card_id=card_id, seed=seed, count=cases_per_card) if real_public else []
    manifest_rows = [
        manifest_row(task, card_id=card_id, source_id=source_id, evidence_level=evidence_level, semantics=semantics, rank=rank)
        for rank, task in enumerate(selected_tasks, start=1)
    ]
    ready = real_public and len(selected_tasks) == cases_per_card and not blockers
    report = {
        "card_id": card_id,
        "source_id": source_id,
        "source_path": real_code.rel_or_abs(source_path),
        "source_exists": source_path.exists(),
        "available_probe_task_count": len(tasks),
        "available_probe_task_count_before_exclusions": unfiltered_task_count,
        "excluded_probe_task_count": unfiltered_task_count - len(tasks),
        "required_task_count": cases_per_card,
        "selected_task_count": len(selected_tasks),
        "benchmark_evidence_level": evidence_level,
        "score_semantics": semantics,
        "ready_for_wide_public_calibration": ready,
        "blockers": blockers,
        "feature_distribution": summarize_tasks(selected_tasks),
        "selection_policy": "rare-combo first, then lowest current feature pressure, stable hash tie-break",
    }
    return report, manifest_rows


def normalize_manifest_paths(values: list[str]) -> list[str]:
    paths: list[str] = []
    for value in values:
        for part in str(value or "").split(","):
            part = part.strip()
            if part:
                paths.append(part)
    return paths


def load_excluded_task_keys(values: list[str]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for path in normalize_manifest_paths(values):
        for card_id, rows in load_case_manifest(path).items():
            for row in rows:
                task_id = str(row.get("task_id") or "").strip()
                if task_id:
                    keys.add((card_id, task_id))
    return keys


def select_diverse_tasks(tasks: list[dict[str, Any]], *, card_id: str, seed: int, count: int) -> list[dict[str, Any]]:
    decorated: list[dict[str, Any]] = []
    for task in tasks:
        features = classify_task(task)
        rank = stable_rank(f"{seed}:{card_id}:{task.get('task_id')}")
        decorated.append({"task": task, "features": features, "rank": rank, "combo": feature_combo(features)})
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decorated:
        groups[row["combo"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (row["rank"], str(row["task"].get("task_id") or "")))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    feature_counts: Counter[str] = Counter()

    for combo, rows in sorted(groups.items(), key=lambda item: (len(item[1]), min(row["rank"] for row in item[1]), item[0])):
        if len(selected) >= count:
            break
        row = rows[0]
        task_id = str(row["task"].get("task_id") or "")
        if task_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(task_id)
        feature_counts.update(feature_tokens(row["features"]))

    while len(selected) < count:
        remaining = [row for row in decorated if str(row["task"].get("task_id") or "") not in selected_ids]
        if not remaining:
            break
        row = min(
            remaining,
            key=lambda item: (
                sum(feature_counts[token] for token in feature_tokens(item["features"])),
                item["rank"],
                str(item["task"].get("task_id") or ""),
            ),
        )
        selected.append(row)
        selected_ids.add(str(row["task"].get("task_id") or ""))
        feature_counts.update(feature_tokens(row["features"]))

    return [row["task"] for row in selected[:count]]


def manifest_row(
    task: dict[str, Any],
    *,
    card_id: str,
    source_id: str,
    evidence_level: str,
    semantics: str,
    rank: int,
) -> dict[str, Any]:
    features = classify_task(task)
    task_id = str(task.get("task_id") or "")
    source_task_id = str(task.get("source_task_id") or "")
    return {
        "policy": PUBLIC_CASE_MANIFEST_POLICY,
        "task_id": task_id,
        "source_task_id": source_task_id,
        "card_id": card_id,
        "source_id": source_id,
        "case_type": str(task.get("case_type") or ""),
        "entry_point": str(task.get("entry_point") or ""),
        "task_hash": stable_digest("|".join([card_id, source_id, task_id, source_task_id, str(task.get("case_type") or "")])),
        "selection_rank": int(rank),
        "feature_buckets": features,
        "benchmark_evidence_level": evidence_level,
        "score_semantics": semantics,
        **PUBLIC_CASE_MANIFEST_CONTENT_POLICY,
    }


def classify_task(task: dict[str, Any]) -> dict[str, str]:
    prompt = str(task.get("prompt") or "")
    tags = " ".join(str(tag) for tag in task.get("tags", []) if isinstance(task.get("tags"), list))
    text = f"{prompt} {tags} {task.get('case_type') or ''}".lower()
    signature = real_code.visible_task_signature(task)
    args = signature.get("args") if isinstance(signature, dict) else []
    arg_count = len(args) if isinstance(args, list) else 0
    varargs = bool(signature.get("varargs")) if isinstance(signature, dict) else False
    interface_shape = "varargs" if varargs else "no_arg" if arg_count == 0 else "single_arg" if arg_count == 1 else "multi_arg"
    return {
        "interface_shape": interface_shape,
        "return_shape": return_shape(text),
        "parsing_burden": parsing_burden(text),
        "dependency_shape": dependency_shape(text),
        "algorithm_family": algorithm_family(text),
    }


def return_shape(text: str) -> str:
    if any(token in text for token in ["boolean", " bool", "true or false", "whether ", " return true", " return false", "is valid", "is prime", "palindrome"]):
        return "bool"
    if any(token in text for token in ["dictionary", " dict", "mapping", "frequency", "counter"]):
        return "dict"
    if any(token in text for token in ["tuple", "pair", "coordinates"]):
        return "tuple"
    if any(token in text for token in ["list", "array", "sequence", "return all", "return unique", "filter"]):
        return "list"
    if any(token in text for token in ["string", "substring", "word", "sentence", "character", "binary representation"]):
        return "string"
    if any(token in text for token in ["float", "average", "mean", "ratio", "area", "volume", "distance"]):
        return "float"
    if any(token in text for token in ["integer", "number", "count", "length", "index", "maximum", "minimum", "sum", "product"]):
        return "int"
    return "any"


def parsing_burden(text: str) -> str:
    if any(token in text for token in ["utf", "ascii", "unicode", "base64", "hex", "encode", "decode", "binary"]):
        return "encoding"
    if any(token in text for token in ["json", "csv", "xml", "html", "matrix", "grid", "table"]):
        return "structured_io"
    if any(token in text for token in ["regular expression", "regex", "pattern match", "matches pattern"]):
        return "regex_like"
    if any(token in text for token in ["parse", "split", "extract", "format", "tokenize", "bracket", "parentheses"]):
        return "string_parse"
    if any(token in text for token in ["file", "path", "directory", "filesystem"]):
        return "file_or_system"
    return "none"


def dependency_shape(text: str) -> str:
    if any(token in text for token in ["file", "path", "directory", "os.", "filesystem"]):
        return "filesystem"
    if any(token in text for token in ["datetime", "date", "time zone"]):
        return "datetime"
    if any(token in text for token in ["permutation", "combination", "cartesian", "itertools"]):
        return "itertools"
    if any(token in text for token in ["counter", "frequency", "deque", "defaultdict", "collections"]):
        return "collections"
    if any(token in text for token in ["math", "sqrt", "gcd", "lcm", "prime", "factorial", "log", "trigonometric"]):
        return "stdlib_math"
    if any(token in text for token in ["numpy", "pandas", "dataframe", "requests"]):
        return "external_or_data"
    return "none"


def algorithm_family(text: str) -> str:
    if any(token in text for token in ["graph", "node", "edge", "shortest path", "tree"]):
        return "graph"
    if any(token in text for token in ["dynamic programming", "subsequence", "knapsack", "minimum cost", "ways to"]):
        return "dynamic_programming"
    if parsing_burden(text) not in {"none", "file_or_system"}:
        return "parsing"
    if any(token in text for token in ["empty", "boundary", "duplicate", "negative", "none", "null"]):
        return "edge_case"
    if any(token in text for token in ["type", "convert", "cast", "normalize"]):
        return "type_handling"
    if any(token in text for token in ["sort", "order", "rank", "priority"]):
        return "sorting"
    if any(token in text for token in ["find", "search", "closest", "maximum", "minimum"]):
        return "search"
    if any(token in text for token in ["string", "substring", "word", "character", "palindrome"]):
        return "string"
    if dependency_shape(text) == "stdlib_math" or any(token in text for token in ["sum", "product", "number", "integer"]):
        return "math"
    if any(token in text for token in ["list", "array", "tuple", "dictionary", "filter", "merge"]):
        return "collection_transform"
    if any(token in text for token in ["class ", "method", "api", "object", "system"]):
        return "execution_shaped_program"
    return "general_algorithm"


def feature_combo(features: dict[str, str]) -> str:
    return "|".join(f"{key}={features.get(key, 'unknown')}" for key in FEATURE_KEYS)


def feature_tokens(features: dict[str, str]) -> list[str]:
    return [f"{key}:{features.get(key, 'unknown')}" for key in FEATURE_KEYS]


def summarize_tasks(tasks: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = {key: Counter() for key in FEATURE_KEYS}
    for task in tasks:
        features = classify_task(task)
        for key in FEATURE_KEYS:
            counters[key][features.get(key, "unknown")] += 1
    return {key: dict(sorted(counter.items())) for key, counter in counters.items()}


def summarize_manifest_features(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = {key: Counter() for key in FEATURE_KEYS}
    for row in rows:
        features = row.get("feature_buckets") if isinstance(row.get("feature_buckets"), dict) else {}
        for key in FEATURE_KEYS:
            counters[key][str(features.get(key) or "unknown")] += 1
    return {key: dict(sorted(counter.items())) for key, counter in counters.items()}


def stable_rank(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Wide Public Slice Selector",
        "",
        f"Trigger state: `{report.get('trigger_state')}`",
        f"Ready for wide public calibration: `{report.get('ready_for_wide_public_calibration')}`",
        f"Selected cases: `{report.get('selected_total_cases')}/{report.get('required_total_cases')}`",
        "",
        "Public content policy: prompts/tests/solutions/candidate code are not emitted by this manifest.",
        "",
        "## Cards",
        "",
    ]
    for row in report.get("cards_report", []):
        blockers = row.get("blockers") if isinstance(row.get("blockers"), list) else []
        lines.append(
            f"- `{row.get('card_id')}`: selected `{row.get('selected_task_count')}/{row.get('required_task_count')}`, "
            f"evidence `{row.get('benchmark_evidence_level')}`, ready `{row.get('ready_for_wide_public_calibration')}`"
        )
        for blocker in blockers:
            lines.append(f"  - blocker `{blocker.get('blocker')}`: {blocker.get('detail') or blocker}")
    lines.extend(["", "## Next Actions", ""])
    if report.get("ready_for_wide_public_calibration"):
        lines.append("- Use this manifest only after private gates are GREEN and spend exactly one wide public calibration run.")
    else:
        lines.append("- Stage missing real benchmark payloads or lower the declared target before running public calibration.")
        lines.append("- Do not count loader-regression or metadata-only fallback cases toward the 160-task public proof.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
