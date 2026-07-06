"""Support, scoring, and report helpers for real-code benchmark graduation."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from real_code_benchmark_constants import *  # noqa: F403


__all__ = [
    "write_transfer_artifact",
    "merge_transfer_index",
    "load_transfer",
    "resolve_source_path",
    "behavior_changed",
    "same_task_ids",
    "without_traces",
    "residual",
    "classify_failure",
    "suggested_intervention",
    "repair_pattern",
    "unavailable_external_imports",
    "tags_from_text",
    "function_name",
    "function_name_from_tests",
    "mbpp_visible_signature",
    "function_args",
    "function_signature",
    "visible_task_signature",
    "candidate_body",
    "placeholder_scaffold_code",
    "formula_like_task",
    "candidate_quality",
    "indent_completion",
    "bogus_return_attribute_body",
    "bogus_return_local_callable_body",
    "dedupe_candidates",
    "rotate",
    "ratio",
    "bounded_verification_workers",
    "gate",
    "sha256_text",
    "safe_name",
    "resolve",
    "rel",
    "rel_or_abs",
    "runtime_tmp_dir",
    "read_json",
    "get_path",
    "truthy",
    "read_jsonl",
    "write_json",
    "write_jsonl",
    "now",
]

def write_transfer_artifact(
    path: Path,
    *,
    suites: list[dict[str, Any]],
    transfer: dict[str, Any],
    trace_path: Path,
) -> Path:
    clusters: dict[str, dict[str, Any]] = {}
    repair_traces = []
    for suite in suites:
        for row in suite.get("residuals", []) if isinstance(suite.get("residuals"), list) else []:
            category = str(row.get("type") or "code_repair_failure")
            item = clusters.setdefault(
                category,
                {
                    "category": category,
                    "count": 0,
                    "cards": set(),
                    "examples": [],
                    "suggested_intervention": suggested_intervention(category),
                },
            )
            item["count"] += 1
            item["cards"].add(str(row.get("card_id") or suite.get("card_id") or ""))
            if len(item["examples"]) < 5:
                item["examples"].append(row)
            repair_traces.append(
                {
                    "trace_id": f"real_code_grad_{safe_name(str(row.get('task_id') or row.get('card_id')))}",
                    "created_utc": now(),
                    "card_id": row.get("card_id") or suite.get("card_id"),
                    "task_id": row.get("task_id"),
                    "category": category,
                    "residual_type": category,
                    "repair_pattern": repair_pattern(category),
                    "transfer_hint": f"Load {repair_pattern(category)} before next code-family run and compare pass-rate delta.",
                    "loads_into": ["code_repair_arm", "pressure_runner", "benchmark_adapter_factory", "octopus_router"],
                }
            )
    normalized_clusters = []
    for item in clusters.values():
        normalized_clusters.append(
            {
                **item,
                "cards": sorted(card for card in item["cards"] if card),
                "priority": round(float(item["count"]) + 2.0, 3),
            }
        )
    payload = {
        "policy": "project_theseus_real_code_benchmark_graduation_transfer_artifact_v1",
        "created_utc": now(),
        "family": "coding_local_sandbox",
        "card_id": "real_code_benchmark_graduation",
        "active_card": True,
        "summary": {
            "suite_count": len(suites),
            "cluster_count": len(normalized_clusters),
            "trace_count": len(repair_traces),
            "loaded_prior_transfer_artifacts": len(transfer["artifacts"]),
        },
        "failure_clusters": normalized_clusters,
        "repair_traces": repair_traces,
        "synthesized_tests": [
            {
                "category": "real_code_graduation_gate",
                "name": "real_code_same_case_transfer_delta",
                "purpose": "Require single-stream and multi-stream code runners to score the same public/local task IDs from student checkpoint candidates and export residuals.",
                "template": "assert same_cases and token_level_code_generation_learned and benchmark_promotion_integrity_valid",
                "source_cards": [suite.get("card_id") for suite in suites],
                "risk": "low",
            }
        ],
        "prompt_program_sketches": [
            {
                "category": "real_code_graduation_gate",
                "sketch": "Load public/local benchmark task, consume candidates emitted by a local Theseus checkpoint, sandbox tests, classify residuals, and export transfer evidence without synthesizing benchmark-specific solver code in the harness.",
                "expected_effect": "turn public code benchmarks into honest student pressure and reusable residual traces",
                "loads_into": ["code_repair_arm", "octopus_router", "benchmaxx_curriculum"],
            }
        ],
        "loads_into": ["code_repair_arm", "benchmark_adapter_factory", "pressure_runner", "octopus_router"],
        "verification": {
            "external_inference_calls": 0,
            "student_growth": "required_for_public_code_progress",
            "loop_closure_tool_distillation_allowed": False,
            "trace": rel(trace_path),
        },
    }
    write_json(path, payload)
    return path


def merge_transfer_index(index_path: Path, artifact_path: Path) -> None:
    index = read_json(index_path, {})
    artifacts = [row for row in index.get("artifacts", []) if isinstance(row, dict)] if isinstance(index.get("artifacts"), list) else []
    rel_path = rel(artifact_path)
    artifacts = [row for row in artifacts if str(row.get("path") or "") != rel_path]
    payload = read_json(artifact_path, {})
    artifacts.append(
        {
            "name": "real_code_benchmark_graduation_transfer",
            "family": "coding_local_sandbox",
            "card_id": "real_code_benchmark_graduation",
            "path": rel_path,
            "loads_into": payload.get("loads_into") or ["code_repair_arm", "pressure_runner"],
            "cluster_count": len(payload.get("failure_clusters", [])) if isinstance(payload.get("failure_clusters"), list) else 0,
            "trace_count": len(payload.get("repair_traces", [])) if isinstance(payload.get("repair_traces"), list) else 0,
            "active_card": True,
        }
    )
    write_json(
        index_path,
        {
            "policy": "project_theseus_code_transfer_artifacts_index_v1",
            "created_utc": now(),
            "summary": {
                "frontier_family": "coding_local_sandbox",
                "active_card_id": "real_code_benchmark_graduation",
                "artifact_count": len(artifacts),
                "cluster_count": sum(int(row.get("cluster_count") or 0) for row in artifacts),
                "trace_count": sum(int(row.get("trace_count") or 0) for row in artifacts),
                "loads_into": ["code_repair_arm", "benchmark_adapter_factory", "pressure_runner", "octopus_router"],
            },
            "artifacts": artifacts,
            "external_inference_calls": 0,
        },
    )


def load_transfer(path: Path) -> dict[str, Any]:
    index = read_json(path, {})
    artifacts = []
    categories: list[str] = []
    for item in index.get("artifacts", []) if isinstance(index.get("artifacts"), list) else []:
        if not isinstance(item, dict):
            continue
        artifact_path = resolve(str(item.get("path") or ""))
        payload = read_json(artifact_path, {})
        if not payload:
            continue
        artifacts.append({"path": rel_or_abs(artifact_path), "card_id": item.get("card_id")})
        for cluster in payload.get("failure_clusters", []) if isinstance(payload.get("failure_clusters"), list) else []:
            category = str(cluster.get("category") or "")
            if category and category not in categories:
                categories.append(category)
    if not categories:
        categories = ["repair_loop", "edge_case", "type_handling", "algorithm_choice", "parsing"]
    return {
        "policy": "project_theseus_real_code_transfer_consumption_v1",
        "source": rel_or_abs(path),
        "artifacts": artifacts,
        "categories": categories,
    }


def resolve_source_path(card: dict[str, Any]) -> Path:
    source_id = safe_name(str(card.get("source_id") or "") or str(card.get("id") or "").replace("source_", ""))
    local_roots = [
        os.environ.get("THESEUS_PUBLIC_BENCHMARK_ROOT", ""),
        os.environ.get("THESEUS_RESOURCE_PANTRY_ROOT", ""),
        "resource_pantry/git",
        "data/public_benchmark_sources",
        "data/benchmark_sources",
    ]
    first_local_candidate: Path | None = None
    for root in local_roots:
        if not root or not source_id:
            continue
        for leaf in [source_id, source_id.replace("_", "-")]:
            path = resolve(str(Path(root) / leaf))
            if first_local_candidate is None:
                first_local_candidate = path
            if path.exists():
                return path
    candidates = [
        str(card.get("resource_pantry_path") or ""),
        str(card.get("staged_path") or ""),
    ]
    for candidate in list(candidates):
        if not candidate:
            continue
        match = re.search(r"resource_pantry[/\\](?:git|external_benchmark_candidates)[/\\](.+)$", candidate)
        if not match:
            continue
        relative = match.group(1).replace("\\", "/")
        for root in local_roots:
            if not root:
                continue
            path = resolve(str(Path(root) / relative))
            if path.exists():
                return path
    for candidate in candidates:
        if not candidate:
            continue
        path = resolve(candidate)
        if path.exists():
            return path
    if first_local_candidate is not None:
        return first_local_candidate
    return resolve(candidates[0] if candidates and candidates[0] else "")


def behavior_changed(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> bool:
    before_by_id = {str(row.get("task_id")): row for row in before}
    for row in after:
        prev = before_by_id.get(str(row.get("task_id")))
        if not prev:
            continue
        if bool(prev.get("passed")) != bool(row.get("passed")):
            return True
        if prev.get("candidate_sha256") != row.get("candidate_sha256"):
            return True
    return False


def same_task_ids(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> bool:
    return {str(row.get("task_id")) for row in before} == {str(row.get("task_id")) for row in after}


def without_traces(suites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for suite in suites:
        out.append({key: value for key, value in suite.items() if key != "traces"})
    return out


def residual(kind: str, message: str, *, card_id: str, task_id: Any = None, detail: str = "") -> dict[str, Any]:
    return {"type": kind, "card_id": card_id, "task_id": task_id, "message": message, "detail": detail[:1000]}


def classify_failure(stderr: Any) -> str:
    text = str(stderr or "").lower()
    if "lint_parse_failed" in text or "candidate_compile_failed" in text or "test_harness_compile_failed" in text:
        return "verification_cascade_compile"
    if "candidate_dependency_unavailable" in text or "unavailable_external_import" in text or "modulenotfounderror" in text:
        return "external_dependency_missing"
    if "runtime_failed" in text:
        return "runtime_load_failure"
    if "sandbox_launch_failed" in text:
        return "verification_sandbox_launch"
    if "beautiful_code_quality_gate_failed" in text:
        return "code_quality_gate"
    if "missing local theseus student checkpoint candidate" in text:
        return "local_code_generation_adapter_needed"
    if "syntaxerror" in text or "indentation" in text:
        return "parsing"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "typeerror" in text or "attributeerror" in text or "nameerror" in text:
        return "type_handling"
    if "assert" in text:
        return "edge_case"
    return "algorithm_choice"


def suggested_intervention(category: str) -> str:
    return {
        "parsing": "Tighten candidate extraction and syntax repair before sandbox execution.",
        "timeout": "Chunk tests and emit resumable partial evidence before retry.",
        "type_handling": "Add type/null/shape guards to code repair candidates.",
        "edge_case": "Synthesize boundary tests from failed public assertions.",
        "algorithm_choice": "Route to algorithm-template selection and complexity probes.",
        "verification_cascade_compile": "Keep failed candidates out of sandbox execution and train on lint/compile residuals first.",
        "runtime_load_failure": "Reward compile success separately, then target import/name/type runtime-load residuals before semantic tests.",
        "external_dependency_missing": "Prefer stdlib/local implementations or explicitly available dependencies before sandbox runtime.",
        "verification_sandbox_launch": "Repair the benchmark sandbox launcher before counting candidate quality.",
        "code_quality_gate": "Prefer exact-signature, non-vacuous, obligation-matching candidates before counting public passes.",
        "local_code_generation_adapter_needed": "Wire an actual local Theseus student checkpoint/code generator into the candidate manifest contract.",
        "source_not_staged": "Skip unstaged code cards until local source is present.",
        "problem_manifest_locator": "Add or refresh local task dataset/manifest locator.",
    }.get(category, "Classify and replay as a deterministic code repair residual.")


def repair_pattern(category: str) -> str:
    return {
        "parsing": "syntax-first patch beam",
        "timeout": "resumable test budget",
        "type_handling": "typed guard synthesis",
        "edge_case": "boundary test synthesis",
        "algorithm_choice": "algorithm template selection",
        "verification_cascade_compile": "lint-compile cascade repair",
        "runtime_load_failure": "runtime-load repair",
        "external_dependency_missing": "dependency-free candidate reranking",
        "verification_sandbox_launch": "sandbox launcher repair",
        "code_quality_gate": "beautiful-code candidate reranking",
        "local_code_generation_adapter_needed": "student checkpoint candidate adapter",
        "source_not_staged": "runnable source gating",
        "problem_manifest_locator": "task manifest locator",
    }.get(category, "local repair replay")


def tags_from_text(text: str) -> list[str]:
    lowered = text.lower()
    tags = ["repair_loop"]
    if any(token in lowered for token in ["empty", "none", "zero", "negative", "edge"]):
        tags.append("edge_case")
    if any(token in lowered for token in ["list", "string", "integer", "type", "tuple"]):
        tags.append("type_handling")
    if any(token in lowered for token in ["sort", "search", "minimum", "maximum", "sum", "count"]):
        tags.append("algorithm_choice")
    return sorted(set(tags))


def function_name(code: str) -> str:
    match = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code, re.M)
    return match.group(1) if match else ""


def function_name_from_tests(tests: Any, preferred: str = "") -> str:
    wrapper_calls = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
    }
    preferred = str(preferred or "")
    for item in tests if isinstance(tests, list) else []:
        names = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(item))
        if preferred and preferred in names:
            return preferred
        for name in names:
            if name not in wrapper_calls and name != "assert":
                return name
    return ""


def mbpp_visible_signature(entry: str, task_text: str) -> str:
    """Build a visible MBPP signature from prompt text only.

    Public tests and reference implementations stay hidden from generation.
    The goal is to avoid erasing the interface into bare ``*args`` while still
    preserving enough fallback for prompts with unclear arity.
    """

    name = re.sub(r"[^A-Za-z0-9_]", "_", entry or "solve")
    if not re.match(r"^[A-Za-z_]", name):
        name = f"solve_{name}"
    text = f" {task_text.lower()} "
    three_arg_needles = [
        " three ",
        " 3 ",
        "three numbers",
        "three lists",
        "three tuples",
        "three strings",
        "triangle",
    ]
    two_arg_needles = [
        " two ",
        " 2 ",
        "two lists",
        "two tuples",
        "two strings",
        "two numbers",
        "two dictionaries",
        "first and second",
        "given a and b",
        "given x and y",
    ]
    if any(token in text for token in three_arg_needles):
        args = "data, other, third"
    elif any(token in text for token in two_arg_needles):
        args = "data, other"
    else:
        args = "data"
    return f"def {name}({args}):"


def function_args(code: str) -> list[str]:
    match = re.search(r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(([^)]*)\)", code, re.M)
    if not match:
        return []
    args = []
    for raw in match.group(1).split(","):
        name = raw.strip().split(":", 1)[0].split("=", 1)[0].strip()
        if name and name not in {"self", "*"} and not name.startswith("*"):
            args.append(name)
    return args


def function_signature(code: str) -> dict[str, Any]:
    match = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", code, re.M)
    if not match:
        return {"name": "", "args": [], "varargs": False}
    args = []
    varargs = False
    for raw in match.group(2).split(","):
        item = raw.strip()
        if not item or item in {"/", "*"}:
            continue
        if item.startswith("*"):
            varargs = True
            continue
        name = item.split(":", 1)[0].split("=", 1)[0].strip()
        if name and name != "self":
            args.append(name)
    return {"name": match.group(1), "args": args, "varargs": varargs}


def visible_task_signature(task: dict[str, Any]) -> dict[str, Any]:
    prompt = str(task.get("prompt") or "")
    entry = str(task.get("entry_point") or "")
    for match in re.finditer(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", prompt, re.M):
        if not entry or match.group(1) == entry:
            return function_signature(match.group(0))
    return {"name": entry, "args": [], "varargs": False}


def candidate_body(code: str) -> str:
    lines = code.splitlines()
    out: list[str] = []
    in_body = False
    for line in lines:
        if not in_body:
            if re.match(r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", line):
                in_body = True
            continue
        if line.strip() and not line.startswith((" ", "\t")):
            break
        out.append(line[4:] if line.startswith("    ") else line.lstrip("\t"))
    return "\n".join(out)


def placeholder_scaffold_code(code: str) -> bool:
    lowered = code.lower()
    compact = re.sub(r"\s+", "", lowered)
    if any(
        token in compact
        for token in [
            "args=(",
            "data=args[",
            "other=args[",
            "extra=args[",
            "extra=()",
        ]
    ):
        return True
    if re.search(r"^\s*(args|extra)\s*=", code, re.M):
        return True
    if re.search(r"^\s*other\s*=\s*None\s*$", code, re.M):
        return True
    return bool(
        re.search(r"^\s*result\s*=\s*(False|True|None|0|1|\[\]|\{\}|\(\)|''|\"\")\s*$", code, re.M)
        and re.search(r"^\s*return\s+result\s*$", code, re.M)
        and not re.search(r"result\s*(\+=|\.append|\.extend|\.update)", code)
    )


def formula_like_task(task: dict[str, Any]) -> bool:
    text = f"{task.get('category') or ''} {task.get('prompt') or ''} {' '.join(map(str, task.get('tags') or []))}".lower()
    return any(
        token in text
        for token in [
            "area",
            "volume",
            "surface",
            "perimeter",
            "circumference",
            "simple interest",
            "compound interest",
            "fahrenheit",
            "celsius",
            "distance",
            "mean",
            "average",
            "sum",
            "product",
            "multiply",
            "divide",
            "modulo",
            "power",
            "square",
            "cube",
            "factorial",
            "gcd",
            "lcm",
            "prime",
            "woodall",
        ]
    )


def candidate_quality(task: dict[str, Any], code: str, origin: str = "") -> dict[str, Any]:
    sig = function_signature(code)
    expected = visible_task_signature(task)
    body = candidate_body(code)
    lowered_body = body.lower()
    reasons: list[str] = []
    score = 0.0

    if not sig["name"]:
        reasons.append("missing_function_def")
        score -= 8.0
    elif expected["name"] and sig["name"] != expected["name"]:
        reasons.append("entry_point_mismatch")
        score -= 5.0
    else:
        score += 1.5

    if sig["varargs"]:
        reasons.append("erased_varargs_signature")
        score -= 8.0
    elif expected["args"] and sig["args"] != expected["args"]:
        reasons.append("visible_signature_args_mismatch")
        score -= 3.0
    else:
        score += 1.3

    if placeholder_scaffold_code(code):
        reasons.append("placeholder_scaffold_body")
        score -= 8.0
    else:
        score += 1.2

    meaningful_lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not meaningful_lines:
        reasons.append("empty_body")
        score -= 5.0
    elif len(meaningful_lines) >= 2:
        score += 0.4

    if "return" not in lowered_body:
        reasons.append("missing_return")
        score -= 4.0
    else:
        score += 0.8

    if formula_like_task(task):
        if re.search(r"(\+|-|\*|/|%|\*\*|pow\(|math\.)", body):
            score += 1.4
        else:
            reasons.append("formula_task_without_arithmetic_obligation")
            score -= 4.5

    collection_like = any(token in str(task.get("prompt") or "").lower() for token in ["list", "array", "tuple", "string", "dict"])
    transform_like = any(token in str(task.get("prompt") or "").lower() for token in ["sort", "filter", "count", "remove", "find", "convert", "replace", "merge", "split"])
    if collection_like and transform_like:
        if re.search(r"\b(for|while)\b|\.append\(|\.extend\(|sorted\(|sum\(|len\(|Counter\(|\.split\(", body):
            score += 0.9
        else:
            reasons.append("collection_transform_without_construct")
            score -= 1.8

    if re.search(r"^\s*return\s+(data|args|other|extra)\s*$", body, re.M) and not any(
        token in str(task.get("prompt") or "").lower() for token in ["return the given", "return a given", "identity"]
    ):
        reasons.append("vacuous_identity_return")
        score -= 3.5
    if bogus_return_attribute_body(body):
        reasons.append("bogus_return_attribute")
        score -= 6.0
    if bogus_return_local_callable_body(body):
        reasons.append("bogus_return_local_callable")
        score -= 6.0
    missing_imports = unavailable_external_imports(code)
    if missing_imports:
        for module in missing_imports[:4]:
            reasons.append(f"unavailable_external_import:{module}")
        score -= 5.0 + min(4, len(missing_imports))

    mode = candidate_mode_from_origin(origin)
    if "edge_exec_sparse_state_sequence_sts" in mode:
        score += 1.1
    elif "private_body_ngram_sts" in mode:
        score += 0.7
    elif "local_adapter_edge_skeleton" in mode and "placeholder_scaffold_body" in reasons:
        score -= 2.0

    hard_reasons = {
        "missing_function_def",
        "entry_point_mismatch",
        "erased_varargs_signature",
        "placeholder_scaffold_body",
        "bogus_return_attribute",
        "bogus_return_local_callable",
        "formula_task_without_arithmetic_obligation",
        "empty_body",
        "missing_return",
    }
    hard_reason_prefixes = ("unavailable_external_import:",)
    return {
        "beautiful_code_score": round(score, 6),
        "beautiful_code_reasons": reasons,
        "beautiful_code_gate_passed": score >= 1.0
        and not any(
            reason in hard_reasons or any(reason.startswith(prefix) for prefix in hard_reason_prefixes)
            for reason in reasons
        ),
    }


def candidate_mode_from_origin(origin: str) -> str:
    parts = origin.split(":")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return origin or "unknown_candidate_origin"


def unavailable_external_imports(source: str) -> list[str]:
    """Return unguarded imported top-level modules absent from the sandbox runtime.

    Guarded optional imports are allowed to reach runtime verification so their
    pure-Python fallback path can be judged. Unguarded imports are still blocked
    before sandbox execution.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    stdlib.update({"__future__", "typing"})
    missing: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module.split(".")[0]]
        for name in names:
            if not name or name in stdlib or name.startswith("_"):
                continue
            try:
                available = importlib.util.find_spec(name) is not None
            except (ImportError, ValueError):
                available = False
            if not available and not import_is_guarded_by_try(node, parents) and name not in missing:
                missing.append(name)
    return missing


def import_is_guarded_by_try(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.Try) and any(handler_catches_import_failure(handler) for handler in current.handlers):
            return True
    return False


def handler_catches_import_failure(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names: list[str] = []
    if isinstance(handler.type, ast.Name):
        names.append(handler.type.id)
    elif isinstance(handler.type, ast.Attribute):
        names.append(handler.type.attr)
    elif isinstance(handler.type, ast.Tuple):
        for item in handler.type.elts:
            if isinstance(item, ast.Name):
                names.append(item.id)
            elif isinstance(item, ast.Attribute):
                names.append(item.attr)
    return any(name in {"BaseException", "Exception", "ImportError", "ModuleNotFoundError"} for name in names)


def indent_completion(text: str) -> str:
    return "\n" + "\n".join("    " + line if line else "" for line in text.splitlines()) + "\n"


def bogus_return_attribute_body(body: str) -> bool:
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("return "):
            continue
        expr = re.sub(r"\s+", "", stripped[len("return "):].lower())
        if any(token in expr for token in "([{"):
            continue
        match = re.fullmatch(r"[a-z_][a-z0-9_]*\.([a-z_][a-z0-9_]*)", expr)
        if not match:
            continue
        if match.group(1) in {
            "isinstance",
            "list",
            "dict",
            "tuple",
            "str",
            "int",
            "float",
            "bool",
            "set",
            "len",
            "sum",
            "min",
            "max",
            "sorted",
            "range",
            "append",
            "extend",
            "insert",
            "remove",
            "pop",
            "sort",
            "reverse",
            "items",
            "keys",
            "values",
            "get",
            "split",
            "strip",
            "lower",
            "upper",
            "replace",
            "join",
        }:
            return True
    return False


def bogus_return_local_callable_body(body: str) -> bool:
    allowed_callables = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "pow",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False
    for fn in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
        assigned: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name):
                assigned.add(node.target.id)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.optional_vars, ast.Name):
                        assigned.add(item.optional_vars.id)
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in assigned
                and node.value.func.id not in allowed_callables
            ):
                return True
    return False


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for candidate in candidates:
        code = str(candidate.get("code") or "")
        if not code:
            continue
        digest = sha256_text(code)
        if digest in seen:
            continue
        seen.add(digest)
        row = dict(candidate)
        row["origin"] = str(candidate.get("origin") or "candidate")
        row["code"] = code
        out.append(row)
    return out[:12]


def rotate(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    offset = seed % len(rows)
    return rows[offset:] + rows[:offset]


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def fraction(num: int, den: int) -> str:
    return f"{int(num)}/{int(den)}"


def wilson_ci(num: int, den: int, z: float = 1.959963984540054) -> dict[str, Any]:
    num = int(num)
    den = int(den)
    if den <= 0:
        return {"count": num, "denominator": den, "low": 0.0, "high": 0.0}
    p = num / den
    z2 = z * z
    denom = 1.0 + z2 / den
    center = (p + z2 / (2.0 * den)) / denom
    margin = z * ((p * (1.0 - p) / den + z2 / (4.0 * den * den)) ** 0.5) / denom
    return {
        "count": num,
        "denominator": den,
        "low": round(max(0.0, center - margin), 6),
        "high": round(min(1.0, center + margin), 6),
    }


def bounded_verification_workers(requested: int) -> int:
    if requested > 0:
        return max(1, min(requested, 32))
    env_value = os.environ.get("THESEUS_CODE_VERIFY_WORKERS", "").strip()
    if env_value:
        try:
            return max(1, min(int(env_value), 32))
        except ValueError:
            pass
    cores = os.cpu_count() or 4
    return max(2, min(12, cores))


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "item")).strip("_") or "item"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def rel_or_abs(path: str | Path) -> str:
    return rel(path)


def runtime_tmp_dir() -> Path:
    if os.name == "nt":
        preferred = Path("D:/ProjectTheseus/tmp")
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            return preferred
        except OSError:
            pass
    fallback = ROOT / "reports" / "tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def read_json(path: Path, default: Any = None) -> Any:
    default = {} if default is None else default
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
