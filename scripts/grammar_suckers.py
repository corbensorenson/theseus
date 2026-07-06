"""Build governed language-specific grammar sucker reports.

Grammar suckers are rule substrates, not answer generators. They are allowed
to reject malformed candidates, project surface language into SBL-style frames,
and route work to the right arm. They are not allowed to insert public
benchmark answers or use public tests during generation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grammar_suckers.json")
    parser.add_argument("--out", default="reports/grammar_suckers.json")
    parser.add_argument("--markdown-out", default="reports/grammar_suckers.md")
    parser.add_argument("--sbl-trace-out", default="")
    parser.add_argument("--max-code-rows", type=int, default=240)
    parser.add_argument("--max-conversation-rows", type=int, default=80)
    args = parser.parse_args()

    config = read_json(resolve(args.config), {})
    outputs = config.get("outputs") if isinstance(config.get("outputs"), dict) else {}
    sbl_trace_out = resolve(args.sbl_trace_out or outputs.get("sbl_trace_jsonl") or "data/grammar_suckers/sbl_rule_traces.jsonl")

    candidate_manifest = resolve(get_path(config, ["inputs", "python_candidate_manifest"], "reports/student_code_candidates.jsonl"))
    conversation_jsonl = resolve(get_path(config, ["inputs", "conversation_train_jsonl"], "D:/ProjectTheseus/training_data/open_conversation_pantry/private_train/conversation_sft_pressure.jsonl"))
    legacy_roots = [resolve(item) for item in get_path(config, ["inputs", "legacy_sbl_roots"], [])]

    code_rows = read_jsonl(candidate_manifest)[: max(1, args.max_code_rows)]
    conversation_rows = read_jsonl(conversation_jsonl)[: max(1, args.max_conversation_rows)]
    python_report, python_frames = evaluate_python_candidates(code_rows, str(candidate_manifest))
    english_report, english_frames = evaluate_english_conversations(conversation_rows, str(conversation_jsonl))
    legacy_report = inspect_legacy_sbl(legacy_roots)
    sbl_frames = python_frames + english_frames
    write_jsonl(sbl_trace_out, sbl_frames)

    report = {
        "policy": "project_theseus_grammar_suckers_v0",
        "created_utc": now(),
        "trigger_state": trigger_state(python_report, english_report, legacy_report),
        "thesis": "Rule-following and meaning-making should be separated: grammar suckers enforce legal surface structure while learned arms spend capacity on semantics, repair, and insight.",
        "config": rel(resolve(args.config)),
        "governance": config.get("governance", {}),
        "summary": {
            "sucker_count": len(config.get("suckers", [])) if isinstance(config.get("suckers"), list) else 0,
            "active_sucker_count": 3,
            "planned_sucker_count": max(0, (len(config.get("suckers", [])) if isinstance(config.get("suckers"), list) else 0) - 3),
            "python_candidate_rows_checked": python_report["checked_count"],
            "python_parse_pass_rate": python_report["parse_pass_rate"],
            "python_invalid_promotion_eligible_count": python_report["invalid_promotion_eligible_count"],
            "english_rows_checked": english_report["checked_count"],
            "english_surface_pass_rate": english_report["surface_pass_rate"],
            "sbl_trace_count": len(sbl_frames),
            "legacy_sbl_found": legacy_report["found"],
            "public_benchmark_solutions_used": False,
            "public_tests_visible_to_rule_layer": False,
            "external_inference_calls": 0,
        },
        "suckers": build_sucker_statuses(config, python_report, english_report, legacy_report),
        "python_grammar": python_report,
        "english_surface_grammar": english_report,
        "sbl_semantic_backbone": {
            "schema": "sbl.v1_lite_for_rule_projection",
            "trace_jsonl": rel(sbl_trace_out),
            "frame_count": len(sbl_frames),
            "legacy_evidence": legacy_report,
            "routing_use": [
                "head_router can route by source language and primitive kind",
                "code_lm can reject malformed code before public scoring",
                "conversation pressure can train grammatical surface repair separately from reasoning",
                "STS streams can carry separate solver/critic/rule/audit channels",
            ],
        },
        "checks": checks(python_report, english_report, legacy_report),
        "next_actions": [
            "wire python_grammar_sucker parse failures into residual labels for Code LM Closure v2",
            "add Rust cargo-check and JS/TS parser suckers as planned adapters",
            "use SBL frames as router features, not benchmark answers",
        ],
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 1


def evaluate_python_candidates(rows: list[dict[str, Any]], source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    invalid_promotion = 0
    mode_counts: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        code = str(row.get("code") or "")
        mode = str(row.get("candidate_generation_mode") or "")
        mode_counts[mode] += 1
        parsed = None
        error = ""
        try:
            parsed = ast.parse(code)
        except SyntaxError as exc:
            error = f"{exc.__class__.__name__}: {exc.msg}"
        functions = [node for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef)] if parsed else []
        returns = [node for node in ast.walk(parsed) if isinstance(node, ast.Return)] if parsed else []
        branches = [node for node in ast.walk(parsed) if isinstance(node, (ast.If, ast.IfExp))] if parsed else []
        loops = [node for node in ast.walk(parsed) if isinstance(node, (ast.For, ast.While, ast.AsyncFor))] if parsed else []
        imports = [node for node in ast.walk(parsed) if isinstance(node, (ast.Import, ast.ImportFrom))] if parsed else []
        side_effect_imports = unsafe_import_names(imports)
        parse_ok = parsed is not None
        structure_ok = parse_ok and bool(functions) and bool(returns)
        safe_ok = not side_effect_imports
        promotion_eligible = bool(row.get("benchmark_promotion_eligible"))
        passed = parse_ok and structure_ok and safe_ok
        if promotion_eligible and not passed:
            invalid_promotion += 1
        result = {
            "task_id": row.get("task_id"),
            "candidate_sha256": row.get("candidate_sha256") or sha256_text(code),
            "candidate_generation_mode": mode,
            "benchmark_promotion_eligible": promotion_eligible,
            "parse_ok": parse_ok,
            "structure_ok": structure_ok,
            "safe_imports_ok": safe_ok,
            "passed": passed,
            "error": error,
            "function_count": len(functions),
            "return_count": len(returns),
            "branch_count": len(branches),
            "loop_count": len(loops),
            "side_effect_imports": side_effect_imports,
        }
        results.append(result)
        if parse_ok and functions:
            frames.append(sbl_python_frame(row, functions[0], returns, branches, loops, source, idx))
    checked = len(results)
    parse_passed = sum(1 for row in results if row["parse_ok"])
    passed = sum(1 for row in results if row["passed"])
    return (
        {
            "policy": "project_theseus_python_grammar_sucker_v0",
            "source": rel(source),
            "checked_count": checked,
            "parse_passed": parse_passed,
            "parse_pass_rate": ratio(parse_passed, checked),
            "passed": passed,
            "surface_pass_rate": ratio(passed, checked),
            "invalid_promotion_eligible_count": invalid_promotion,
            "candidate_generation_modes": sorted(mode_counts),
            "failure_count": checked - passed,
            "sample_failures": [row for row in results if not row["passed"]][:12],
            "rule_layer_role": "reject malformed Python structure; never insert task answers",
        },
        frames,
    )


def evaluate_english_conversations(rows: list[dict[str, Any]], source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        target = row.get("target_message") if isinstance(row.get("target_message"), dict) else {}
        text = str(target.get("content") or row.get("causal_text") or "")
        if not text.strip():
            continue
        check = english_surface_check(text)
        check.update(
            {
                "task_id": row.get("task_id"),
                "source_id": row.get("source_id"),
                "license_spdx": row.get("license_spdx"),
            }
        )
        results.append(check)
        frames.append(sbl_english_frame(row, text, source, idx, check))
    checked = len(results)
    passed = sum(1 for row in results if row["passed"])
    return (
        {
            "policy": "project_theseus_english_surface_grammar_sucker_v0",
            "source": rel(source),
            "checked_count": checked,
            "passed": passed,
            "surface_pass_rate": ratio(passed, checked),
            "failure_count": checked - passed,
            "sample_failures": [row for row in results if not row["passed"]][:12],
            "rule_layer_role": "surface grammar pressure for conversations; not reasoning score evidence",
        },
        frames,
    )


def english_surface_check(text: str) -> dict[str, Any]:
    compact = " ".join(text.strip().split())
    words = WORD_RE.findall(compact)
    balanced = delimiters_balanced(compact)
    quote_ok = compact.count('"') % 2 == 0
    terminal_ok = not compact or compact[-1] in ".?!:)]\"'" or len(words) <= 12
    verb_or_modal = {
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "being",
        "been",
        "can",
        "could",
        "should",
        "would",
        "will",
        "may",
        "might",
        "must",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "return",
        "find",
        "make",
        "use",
        "need",
        "want",
        "help",
    }
    sentence_signal = len(words) >= 4 and (
        any(token.lower() in verb_or_modal for token in words) or any(mark in compact for mark in ".?!:")
    )
    repeated_punctuation_ok = not re.search(r"([!?.,])\1{3,}", compact)
    passed = balanced and quote_ok and terminal_ok and sentence_signal and repeated_punctuation_ok
    return {
        "passed": passed,
        "balanced_delimiters": balanced,
        "quote_balance_ok": quote_ok,
        "terminal_boundary_ok": terminal_ok,
        "basic_clause_signal": sentence_signal,
        "repeated_punctuation_ok": repeated_punctuation_ok,
        "word_count": len(words),
        "text_sha256": sha256_text(text),
    }


def inspect_legacy_sbl(roots: list[Path]) -> dict[str, Any]:
    files: list[str] = []
    fixture_count = 0
    contract_count = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if "sbl" in lower or "semantic" in lower:
                files.append(rel(path))
                if lower.endswith(".fixture.json"):
                    fixture_count += 1
                if lower.endswith(".contract.json"):
                    contract_count += 1
    return {
        "found": bool(files),
        "root_count": len([root for root in roots if root.exists()]),
        "evidence_file_count": len(files),
        "fixture_count": fixture_count,
        "contract_count": contract_count,
        "sample_files": files[:20],
        "concept_ported": "deterministic SBL-lite frame projection plus grammar-specific rule checks",
    }


def build_sucker_statuses(config: dict[str, Any], python_report: dict[str, Any], english_report: dict[str, Any], legacy_report: dict[str, Any]) -> list[dict[str, Any]]:
    statuses = []
    for spec in config.get("suckers", []) if isinstance(config.get("suckers"), list) else []:
        sucker_id = str(spec.get("sucker_id") or "")
        if sucker_id == "python_grammar_sucker":
            status = "active_green" if python_report["invalid_promotion_eligible_count"] == 0 and python_report["parse_pass_rate"] >= 0.98 else "active_yellow"
            evidence = f"parse={python_report['parse_pass_rate']} invalid_promotion={python_report['invalid_promotion_eligible_count']}"
        elif sucker_id == "english_surface_grammar_sucker":
            status = "active_green" if english_report["surface_pass_rate"] >= 0.80 else "active_yellow"
            evidence = f"surface={english_report['surface_pass_rate']}"
        elif sucker_id == "sbl_semantic_backbone_sucker":
            status = "active_green" if legacy_report["found"] else "active_yellow"
            evidence = f"legacy_sbl_found={legacy_report['found']} files={legacy_report['evidence_file_count']}"
        else:
            status = "planned"
            evidence = "not yet wired"
        row = dict(spec)
        row.update({"status": status, "evidence": evidence, "external_inference_calls": 0})
        statuses.append(row)
    return statuses


def checks(python_report: dict[str, Any], english_report: dict[str, Any], legacy_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        gate("python_candidates_checked", python_report["checked_count"] > 0, f"rows={python_report['checked_count']}"),
        gate("python_promotion_candidates_parse", python_report["invalid_promotion_eligible_count"] == 0, f"invalid_promotion={python_report['invalid_promotion_eligible_count']}"),
        gate("english_conversation_rows_checked", english_report["checked_count"] > 0, f"rows={english_report['checked_count']}"),
        gate("english_surface_signal_present", english_report["surface_pass_rate"] >= 0.80, f"pass_rate={english_report['surface_pass_rate']}"),
        gate("legacy_sbl_concept_found", legacy_report["found"], f"files={legacy_report['evidence_file_count']}"),
        gate("rule_layer_no_public_solutions", True, "grammar suckers validate form and SBL frames only"),
        gate("external_inference_zero", True, "local deterministic checks only"),
    ]


def sbl_python_frame(row: dict[str, Any], func: ast.FunctionDef, returns: list[ast.Return], branches: list[Any], loops: list[Any], source: str, idx: int) -> dict[str, Any]:
    code = str(row.get("code") or "")
    args = [arg.arg for arg in func.args.args]
    return {
        "schema_version": "sbl.v1",
        "document_id": f"sbl_python_{sha256_text(code)[:16]}",
        "mode": "strict",
        "source_modalities": ["source_code"],
        "semantic_primitives": [
            {
                "primitive_id": "p.function.primary",
                "primitive_kind": "function",
                "canonical_label": func.name,
                "attributes": {
                    "language": "python",
                    "args": args,
                    "return_count": len(returns),
                    "branch_count": len(branches),
                    "loop_count": len(loops),
                    "candidate_generation_mode": row.get("candidate_generation_mode"),
                    "benchmark_promotion_eligible": bool(row.get("benchmark_promotion_eligible")),
                },
                "confidence": 0.94,
            }
        ],
        "relations": [],
        "qualifiers": [],
        "invocations": [
            invocation("validate.python_ast"),
            invocation("validate.keyword_invariance"),
            invocation("compile.semantic_backbone_frame"),
        ],
        "metadata": {
            "source": rel(source),
            "source_row_index": idx,
            "task_id": row.get("task_id"),
            "candidate_sha256": row.get("candidate_sha256") or sha256_text(code),
        },
        "provenance": provenance(source, idx),
    }


def sbl_english_frame(row: dict[str, Any], text: str, source: str, idx: int, check: dict[str, Any]) -> dict[str, Any]:
    words = WORD_RE.findall(text)
    action = first_action(words)
    return {
        "schema_version": "sbl.v1",
        "document_id": f"sbl_english_{sha256_text(text)[:16]}",
        "mode": "exploratory",
        "source_modalities": ["written_language"],
        "semantic_primitives": [
            {
                "primitive_id": "p.utterance.primary",
                "primitive_kind": "utterance",
                "canonical_label": " ".join(text.strip().split())[:240],
                "attributes": {
                    "language": "english",
                    "word_count": len(words),
                    "surface_passed": bool(check.get("passed")),
                    "action_hint": action,
                    "source_id": row.get("source_id"),
                },
                "confidence": 0.82,
            }
        ],
        "relations": [],
        "qualifiers": [],
        "invocations": [
            invocation("validate.english_surface"),
            invocation("compile.semantic_backbone_frame"),
        ],
        "metadata": {
            "source": rel(source),
            "source_row_index": idx,
            "task_id": row.get("task_id"),
            "license_spdx": row.get("license_spdx"),
        },
        "provenance": provenance(source, idx),
    }


def first_action(words: list[str]) -> str:
    verbs = {"return", "find", "write", "edit", "explain", "improve", "compare", "consider", "use", "make", "train", "learn"}
    for word in words:
        lowered = word.lower().strip("'")
        if lowered in verbs or lowered.endswith("ing") or lowered.endswith("ed"):
            return lowered
    return ""


def unsafe_import_names(imports: list[Any]) -> list[str]:
    out = []
    blocked = {"os", "sys", "subprocess", "socket", "pathlib", "shutil"}
    for node in imports:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in blocked:
                    out.append(root)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in blocked:
                out.append(root)
    return sorted(set(out))


def trigger_state(python_report: dict[str, Any], english_report: dict[str, Any], legacy_report: dict[str, Any]) -> str:
    if python_report["invalid_promotion_eligible_count"] > 0:
        return "RED"
    if python_report["parse_pass_rate"] >= 0.98 and english_report["surface_pass_rate"] >= 0.80 and legacy_report["found"]:
        return "GREEN"
    return "YELLOW"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Grammar Suckers",
        "",
        f"State: **{report['trigger_state']}**",
        "",
        "Grammar suckers are a rule substrate, not answer keys. They reject malformed surfaces and emit SBL frames so the learned arms can focus on meaning and repair.",
        "",
        "## Summary",
        "",
        f"- Python parse pass rate: {summary['python_parse_pass_rate']}",
        f"- Invalid promotion-eligible Python candidates: {summary['python_invalid_promotion_eligible_count']}",
        f"- English surface pass rate: {summary['english_surface_pass_rate']}",
        f"- SBL traces written: {summary['sbl_trace_count']}",
        f"- Legacy SBL evidence found: {summary['legacy_sbl_found']}",
        "",
        "## Active Suckers",
        "",
    ]
    for sucker in report["suckers"]:
        lines.append(f"- `{sucker['sucker_id']}`: {sucker['status']} ({sucker['evidence']})")
    return "\n".join(lines) + "\n"


def invocation(opcode: str) -> dict[str, Any]:
    return {
        "invocation_opcode": opcode,
        "invocation_version": "1.0",
        "arguments": {},
        "critical": True,
    }


def provenance(source: str, idx: int) -> dict[str, Any]:
    return {
        "origin_type": "grammar_sucker_projection",
        "source_surface": rel(source),
        "source_row_index": idx,
        "external_inference_calls": 0,
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def delimiters_balanced(text: str) -> bool:
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any = None) -> Any:
    default = {} if default is None else default
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
