"""Private type-contract diagnostic and Decoder V2 feedback rows.

This is a private-only bridge from residual concept pressure to the code
decoder. It reads generated/private high-transfer rows, infers explicit
return-shape/interface/skeleton contracts from private tests and private
solutions, and emits training rows that Code LM Closure can consume through
its existing high-transfer private-train path.

Public benchmark data is calibration-only. This script may read public summary
reports to decide priority, but it never reads public solutions or public
tests and never writes public benchmark-derived training rows.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PRIVATE_ROOT = Path("D:/ProjectTheseus/training_data/high_transfer/private_train")
DEFAULT_PRIVATE_SOURCES = [
    PRIVATE_ROOT / "type_and_return_shape_residual_code_lm_tasks.jsonl",
    PRIVATE_ROOT / "admissibility_and_interface_residual_code_lm_tasks.jsonl",
    PRIVATE_ROOT / "edge_conditions_residual_code_lm_tasks.jsonl",
    PRIVATE_ROOT / "algorithmic_planning_residual_code_lm_tasks.jsonl",
    PRIVATE_ROOT / "execution_shaped_programs_residual_code_lm_tasks.jsonl",
]
DEFAULT_FEEDBACK_OUT = PRIVATE_ROOT / "type_contract_decoder_feedback.jsonl"
DEFAULT_OUT = REPORTS / "type_contract_diagnostic.json"
DEFAULT_MARKDOWN = REPORTS / "type_contract_diagnostic.md"
PUBLIC_SUMMARY_REPORTS = [
    REPORTS / "broad_transfer_matrix.json",
    REPORTS / "transfer_generalization_audit.json",
    REPORTS / "learning_scoreboard.json",
]


SHAPE_ORDER = ["bool", "list", "dict", "tuple", "str", "number", "none", "unknown"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--private-source",
        action="append",
        default=[],
        help="Private JSONL source. May be passed multiple times.",
    )
    parser.add_argument(
        "--private-sources",
        default="",
        help="Optional semicolon/comma-separated private JSONL sources.",
    )
    parser.add_argument("--max-rows", type=int, default=960)
    parser.add_argument("--feedback-out", default=str(DEFAULT_FEEDBACK_OUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    private_sources = source_paths(args)
    rows, source_stats = load_private_rows(private_sources, max_rows=max(1, int(args.max_rows)))
    feedback_rows, diagnostic_rows = build_feedback_rows(rows)
    write_jsonl(resolve(args.feedback_out), feedback_rows)

    summary = summarize(feedback_rows, diagnostic_rows, source_stats, started=started)
    report = {
        "policy": "project_theseus_type_contract_diagnostic_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if feedback_rows else "YELLOW",
        "summary": summary,
        "private_sources": source_stats,
        "decoder_v2_feedback": decoder_feedback_summary(feedback_rows),
        "dominant_contract_pressure": dominant_pressure(diagnostic_rows),
        "four_card_calibration": {
            "next_command": [
                "python",
                "scripts/broad_transfer_closure_runner.py",
                "--execute",
                "--cards",
                "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                "--seed",
                "14",
                "--max-public-cases-per-card",
                "32",
            ],
            "success_contract": "broad receiver lift across 4 cards without HumanEval regression; public data remains calibration-only",
            "score_semantics": "recommended next calibration command, not executed by this diagnostic",
        },
        "governance": {
            "private_training_only": True,
            "public_summary_reports_read": [rel(path) for path in PUBLIC_SUMMARY_REPORTS if path.exists()],
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_benchmark_training_rows_written": False,
            "teacher_apply_mode": False,
        },
        "improvement_contract": {
            "kind": "new_clean_evidence_produced" if feedback_rows else "useful_failure_residual_captured",
            "evidence": rel(resolve(args.feedback_out)),
            "private_feedback_rows": len(feedback_rows),
            "decoder_v2_contract_rows": sum(1 for row in feedback_rows if isinstance(row.get("decoder_contract"), dict)),
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if feedback_rows else 2


def source_paths(args: argparse.Namespace) -> list[Path]:
    out = [Path(item) for item in args.private_source if str(item).strip()]
    for chunk in str(args.private_sources or "").replace(",", ";").split(";"):
        if chunk.strip():
            out.append(Path(chunk.strip()))
    if not out:
        out = list(DEFAULT_PRIVATE_SOURCES)
    return [resolve_path(path) for path in out]


def load_private_rows(paths: list[Path], *, max_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    source_stats: list[dict[str, Any]] = []
    per_source_rows: list[list[dict[str, Any]]] = []
    seen: set[str] = set()
    for path in paths:
        loaded = 0
        accepted = 0
        missing = not path.exists()
        accepted_rows: list[dict[str, Any]] = []
        for raw in read_jsonl(path):
            loaded += 1
            if not private_row_eligible(raw):
                continue
            task_id = str(raw.get("task_id") or raw.get("source_task_id") or "")
            if not task_id or task_id in seen:
                continue
            item = dict(raw)
            item["_source_jsonl"] = rel(path)
            accepted_rows.append(item)
            seen.add(task_id)
            accepted += 1
        source_stats.append(
            {
                "path": rel(path),
                "exists": not missing,
                "loaded_rows": loaded,
                "accepted_rows": accepted,
                "private_training_rule": "generated/private rows only; public benchmarks calibration-only",
            }
        )
        per_source_rows.append(accepted_rows)
    index = 0
    while len(rows) < max_rows:
        added = False
        for bucket in per_source_rows:
            if index < len(bucket):
                rows.append(bucket[index])
                added = True
                if len(rows) >= max_rows:
                    break
        if not added:
            break
        index += 1
    return rows, source_stats


def private_row_eligible(row: dict[str, Any]) -> bool:
    if row.get("public_benchmark") is not False:
        return False
    if bool(row.get("public_benchmark_solutions_included")):
        return False
    if bool(row.get("public_tests_included")):
        return False
    if not str(row.get("prompt") or "").strip():
        return False
    if not str(row.get("entry_point") or "").strip():
        return False
    if not (str(row.get("solution_body") or "").strip() or str(row.get("solution_expr") or "").strip()):
        return False
    if not str(row.get("license_spdx") or "").strip():
        return False
    return True


def build_feedback_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feedback: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for raw in rows:
        contract = infer_contract(raw)
        body = str(raw.get("solution_body") or "").strip() or body_from_expr(str(raw.get("solution_expr") or ""))
        task_id = safe_name(raw.get("task_id") or raw.get("source_task_id") or raw.get("entry_point"))
        row = {
            "task_id": f"type_contract_feedback_{stable_hash(task_id)[:10]}_{task_id}"[:120],
            "source_task_id": task_id,
            "card_id": "private_type_contract_decoder_feedback",
            "source_id": "local_private_type_contract_diagnostic",
            "split": "train",
            "category": str(raw.get("category") or contract["type_family"] or "type_contract"),
            "prompt": prompt_with_contract(raw, contract),
            "entry_point": safe_name(raw.get("entry_point") or "private_type_contract_func"),
            "solution_expr": str(raw.get("solution_expr") or first_return_expression(body)),
            "solution_body": body,
            "tests": str(raw.get("tests") or ""),
            "tags": feedback_tags(raw, contract),
            "benchmark_evidence_level": "private_type_contract_decoder_feedback_train_only",
            "public_benchmark": False,
            "public_benchmark_solutions_included": False,
            "public_tests_included": False,
            "license_spdx": str(raw.get("license_spdx") or "CC0-1.0"),
            "candidate_expression_eligible": bool(raw.get("candidate_expression_eligible", False)),
            "decoder_contract": contract,
            "provenance": {
                "policy": "project_theseus_type_contract_diagnostic_v1",
                "source_jsonl": raw.get("_source_jsonl"),
                "source_task_id": task_id,
                "public_benchmark_answers_used": False,
                "public_tests_used": False,
                "public_solutions_used": False,
                "derived_from_private_solution_and_private_tests": True,
                "score_semantics": "private decoder contract pressure only",
            },
        }
        feedback.append(row)
        diagnostics.append(
            {
                "source_task_id": task_id,
                "source_jsonl": raw.get("_source_jsonl"),
                "category": row["category"],
                "return_shape": contract["return_shape"],
                "type_family": contract["type_family"],
                "visible_arg_count_hint": contract.get("visible_arg_count_hint"),
                "required_constructs": contract.get("required_constructs", []),
                "residual_label": contract.get("residual_label_hint"),
            }
        )
    return feedback, diagnostics


def infer_contract(row: dict[str, Any]) -> dict[str, Any]:
    category = str(row.get("category") or "")
    prompt = str(row.get("prompt") or "")
    body = str(row.get("solution_body") or "")
    tests = str(row.get("tests") or "")
    expected_shapes = [shape_of_value(value) for value in expected_values_from_tests(tests)]
    return_shape = majority(expected_shapes) or return_shape_from_text(category, prompt, body)
    constructs = required_constructs(row, body)
    type_family = type_family_from_text(category, prompt, body, constructs)
    arg_count = visible_arg_count_from_tests(tests, str(row.get("entry_point") or "")) or arg_count_from_body(body)
    return {
        "policy": "project_theseus_decoder_contract_type_contract_feedback_v1",
        "category": category,
        "return_shape": return_shape,
        "type_family": type_family,
        "visible_arg_count_hint": arg_count,
        "required_constructs": sorted(constructs),
        "residual_label_hint": str(row.get("concept_residual_label") or row.get("residual_concept") or ""),
        "full_body_required": True,
        "guardrail_only": False,
        "feedback_weight": feedback_weight(return_shape, constructs),
        "public_tests_used": False,
        "public_solutions_used": False,
        "score_semantics": "private type/interface/skeleton pressure for Decoder V2",
    }


def expected_values_from_tests(tests: str) -> list[Any]:
    values: list[Any] = []
    for line in tests.splitlines():
        line = line.strip()
        if not line.startswith("assert ") or "==" not in line:
            continue
        expr = line[len("assert ") :]
        try:
            parsed = ast.parse(expr, mode="eval")
        except SyntaxError:
            continue
        node = parsed.body
        if isinstance(node, ast.Compare) and node.ops and isinstance(node.ops[0], ast.Eq) and node.comparators:
            try:
                values.append(ast.literal_eval(node.comparators[0]))
            except (ValueError, SyntaxError):
                continue
    return values


def visible_arg_count_from_tests(tests: str, entry_point: str) -> int | None:
    if not entry_point:
        return None
    for line in tests.splitlines():
        line = line.strip()
        if not line.startswith("assert "):
            continue
        try:
            parsed = ast.parse(line[len("assert ") :], mode="eval")
        except SyntaxError:
            continue
        node = parsed.body
        if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Call):
            continue
        call = node.left
        if isinstance(call.func, ast.Name) and call.func.id == entry_point:
            return len(call.args)
    return None


def arg_count_from_body(body: str) -> int | None:
    names = set(re.findall(r"\b(data|other|extra)\b", body))
    if "extra" in names:
        return 3
    if "other" in names:
        return 2
    if "data" in names:
        return 1
    return None


def shape_of_value(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if value is None:
        return "none"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (int, float)):
        return "number"
    return "unknown"


def majority(values: list[str]) -> str:
    useful = [value for value in values if value != "unknown"]
    if not useful:
        return ""
    counts = Counter(useful)
    return sorted(counts.items(), key=lambda item: (-item[1], SHAPE_ORDER.index(item[0]) if item[0] in SHAPE_ORDER else 99))[0][0]


def return_shape_from_text(category: str, prompt: str, body: str) -> str:
    text = f"{category} {prompt} {body}".lower()
    if any(token in text for token in ["return whether", "returns true", "return true", "return false", "check if", "whether "]):
        return "bool"
    if any(token in text for token in ["dictionary", "dict", "frequency", "counts = {}", "return {"]):
        return "dict"
    if any(token in text for token in ["list", "array", "sequence", "out = []", "append", "return ["]):
        return "list"
    if any(token in text for token in ["tuple", "pair"]):
        return "tuple"
    if any(token in text for token in ["string", "text", "word", "char", "join", "return ''", "return str"]):
        return "str"
    if any(token in text for token in ["count", "sum", "number", "area", "volume", "largest", "smallest", "median"]):
        return "number"
    return "unknown"


def required_constructs(row: dict[str, Any], body: str) -> set[str]:
    prompt = str(row.get("prompt") or "").lower()
    category = str(row.get("category") or "").lower()
    text = f"{category} {prompt} {body}".lower()
    guards = row.get("guardrail_expectations") if isinstance(row.get("guardrail_expectations"), dict) else {}
    constructs: set[str] = set()
    if guards.get("requires_loop") or re.search(r"^\s*(for|while)\s+", body, flags=re.MULTILINE):
        constructs.add("loop")
    if guards.get("requires_branch") or re.search(r"^\s*if\s+", body, flags=re.MULTILINE):
        constructs.add("branch")
    if re.search(r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=", body, flags=re.MULTILINE):
        constructs.add("locals")
    if any(token in text for token in ["count", "frequency", "histogram", "at least"]):
        constructs.add("frequency")
        constructs.add("locals")
    if any(token in text for token in ["select", "best", "smallest", "largest", "minimum", "maximum", "median", "filter"]):
        constructs.add("selection")
        constructs.add("branch")
    if any(token in text for token in ["parse", "split", "suffix", "prefix", "digit", "punctuation", "text"]):
        constructs.add("parsing")
    if any(token in text for token in ["empty", "singleton", "threshold", "boundary", "first", "last", "none"]):
        constructs.add("edge_conditions")
    if any(token in text for token in ["prime", "factor", "gcd", "divisor", "fibonacci", "recurrence", "tribonacci"]):
        constructs.add("algorithmic_planning")
    if is_execution_shaped(category, text):
        constructs.add("execution_shaped_programs")
        constructs.add("locals")
        if any(token in text for token in ["file", "path", "directory", "folder", "zip", "tar", "archive", "log"]):
            constructs.add("file_path")
            constructs.add("branch")
        if any(token in text for token in ["csv", "json", "parse", "payload", "urlencode", "field"]):
            constructs.add("structured_parsing")
        if any(token in text for token in ["csv", "row", "split_", "shuffle"]):
            constructs.add("csv")
            constructs.add("loop")
        if any(token in text for token in ["zip", "tar", "archive", "backup"]):
            constructs.add("archive")
            constructs.add("loop")
        if any(token in text for token in ["system", "platform", "architecture", "memory usage", "subprocess", "process"]):
            constructs.add("system_api")
            constructs.add("branch")
    constructs.add("type_and_return_shape")
    return constructs


def type_family_from_text(category: str, prompt: str, body: str, constructs: set[str]) -> str:
    text = f"{category} {prompt} {body}".lower()
    if "execution_shaped_programs" in constructs or is_execution_shaped(category.lower(), text):
        return "execution_shaped_program"
    if "algorithmic_planning" in constructs:
        return "number_theory_or_recurrence"
    if "parsing" in constructs or any(token in text for token in ["string", "text", "char", "word", "vowel"]):
        return "string_indexing"
    if any(token in text for token in ["list", "array", "tuple", "sequence", "item", "element"]):
        return "collection_logic"
    if any(token in text for token in ["whether", "true", "false", "predicate", "threshold"]):
        return "predicate_logic"
    return "general_semantics"


def is_execution_shaped(category: str, text: str) -> bool:
    if category.startswith("private_exec_") or "execution_shaped_program" in text:
        return True
    markers = [
        "archive",
        "backup",
        "csv file",
        "file path",
        "directory",
        "json",
        "payload",
        "process",
        "subprocess",
        "tar.gz",
        "url-encoded",
        "zip file",
    ]
    return any(marker in text for marker in markers)


def feedback_weight(return_shape: str, constructs: set[str]) -> float:
    weight = 1.0
    if return_shape in {"list", "dict", "tuple", "bool"}:
        weight += 0.25
    if {"loop", "branch", "locals"} & constructs:
        weight += 0.25
    if {"frequency", "selection", "parsing", "algorithmic_planning"} & constructs:
        weight += 0.35
    if "execution_shaped_programs" in constructs:
        weight += 0.45
    return round(weight, 3)


def prompt_with_contract(row: dict[str, Any], contract: dict[str, Any]) -> str:
    base = str(row.get("prompt") or "").strip()
    constructs = ", ".join(contract.get("required_constructs") or [])
    arg_hint = contract.get("visible_arg_count_hint")
    return (
        f"{base}\n\n"
        f"Private decoder contract: return_shape={contract.get('return_shape')}; "
        f"type_family={contract.get('type_family')}; "
        f"visible_arg_count={arg_hint if arg_hint is not None else 'unknown'}; "
        f"required_constructs={constructs or 'none'}."
    ).strip()


def feedback_tags(row: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    tags = [str(tag) for tag in row.get("tags", [])] if isinstance(row.get("tags"), list) else []
    tags.extend(
        [
            "type_contract_decoder_feedback",
            f"return_shape:{contract.get('return_shape')}",
            f"type_family:{contract.get('type_family')}",
        ]
    )
    tags.extend(f"construct:{item}" for item in contract.get("required_constructs") or [])
    return sorted(set(tag for tag in tags if tag))


def summarize(
    feedback_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    source_stats: list[dict[str, Any]],
    *,
    started: float,
) -> dict[str, Any]:
    shape_counts = Counter(str(row.get("return_shape") or "unknown") for row in diagnostic_rows)
    family_counts = Counter(str(row.get("type_family") or "unknown") for row in diagnostic_rows)
    construct_counts: Counter[str] = Counter()
    for row in diagnostic_rows:
        construct_counts.update(str(item) for item in row.get("required_constructs") or [])
    return {
        "private_source_count": len(source_stats),
        "private_sources_existing": sum(1 for row in source_stats if row.get("exists")),
        "input_private_rows": sum(int(row.get("accepted_rows") or 0) for row in source_stats),
        "feedback_rows_written": len(feedback_rows),
        "decoder_contract_rows": sum(1 for row in feedback_rows if isinstance(row.get("decoder_contract"), dict)),
        "return_shape_counts": dict(shape_counts.most_common()),
        "type_family_counts": dict(family_counts.most_common()),
        "construct_counts": dict(construct_counts.most_common()),
        "training_ready": bool(feedback_rows),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def decoder_feedback_summary(feedback_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_shape: dict[str, int] = defaultdict(int)
    by_construct: dict[str, int] = defaultdict(int)
    for row in feedback_rows:
        contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
        by_shape[str(contract.get("return_shape") or "unknown")] += 1
        for construct in contract.get("required_constructs") or []:
            by_construct[str(construct)] += 1
    return {
        "feedback_jsonl": rel(DEFAULT_FEEDBACK_OUT),
        "consumed_by_default_code_lm_closure": True,
        "decoder_path": "visible_prompt -> signature/types -> return_shape -> AST/body skeleton -> branch/loop/local token decode -> test repair",
        "shape_pressure": dict(sorted(by_shape.items())),
        "construct_pressure": dict(sorted(by_construct.items())),
        "sts_conditioning_recommendation": "feed type_contract_decoder_feedback rows before public 4-card calibration; keep public reports calibration-only",
    }


def dominant_pressure(diagnostic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combos: Counter[tuple[str, str, str]] = Counter()
    for row in diagnostic_rows:
        constructs = ",".join(row.get("required_constructs") or [])
        combos[(str(row.get("return_shape")), str(row.get("type_family")), constructs)] += 1
    out = []
    for (shape, family, constructs), count in combos.most_common(12):
        out.append({"return_shape": shape, "type_family": family, "required_constructs": constructs.split(",") if constructs else [], "count": count})
    return out


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Type-Contract Diagnostic",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- feedback_rows_written: `{summary.get('feedback_rows_written')}`",
        f"- decoder_contract_rows: `{summary.get('decoder_contract_rows')}`",
        f"- training_ready: `{summary.get('training_ready')}`",
        "",
        "## Return Shapes",
        "",
    ]
    for shape, count in (summary.get("return_shape_counts") or {}).items():
        lines.append(f"- `{shape}`: `{count}`")
    lines.extend(["", "## Constructs", ""])
    for construct, count in (summary.get("construct_counts") or {}).items():
        lines.append(f"- `{construct}`: `{count}`")
    lines.extend(
        [
            "",
            "## Next Calibration",
            "",
            "`python scripts/broad_transfer_closure_runner.py --execute --cards source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench --seed 14 --max-public-cases-per-card 32`",
            "",
            "Public benchmark artifacts remain calibration-only.",
        ]
    )
    return "\n".join(lines) + "\n"


def body_from_expr(expr: str) -> str:
    expr = expr.strip()
    return f"return {expr}" if expr else "return None"


def first_return_expression(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("return "):
            return stripped[len("return ") :].strip()
    return ""


def safe_name(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "item")).strip("_") or "item"


def stable_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except OSError:
        return []
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
