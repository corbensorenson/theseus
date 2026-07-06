"""Verify local multi-stream trace manifests.

The verifier is intentionally stricter than the runner. It checks that stream
rows preserve causal ordering, that monitoring streams exist, that generated
pressure is quarantined from public comparator claims, and that prompt shapes do
not resemble known multi-stream perturbation attacks.
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="reports/multi_stream_causal_verifier.json")
    args = parser.parse_args()

    policy = read_json(resolve(args.policy), {})
    cases = read_jsonl(resolve(args.manifest))
    case_results = [verify_case(case, policy) for case in cases]
    hard_failures = [
        issue
        for case in case_results
        for issue in case.get("issues", [])
        if issue.get("severity") == "hard"
    ]
    monitorability_coverage = ratio(
        sum(1 for row in case_results if row.get("monitorability_present")),
        len(case_results),
    )
    score = verifier_score(case_results, monitorability_coverage)
    gates = [
        gate("manifest_loaded", bool(cases), f"cases={len(cases)}"),
        gate("strict_past_causality", not any(issue["type"] == "future_dependency" for issue in hard_failures), "all dependencies point to prior rows"),
        gate("stream_schema_consistent", not any(issue["type"] == "stream_schema" for issue in hard_failures), "required streams present"),
        gate("public_comparator_quarantined", not any(issue["type"] == "public_comparator" for issue in hard_failures), "public comparator forbidden"),
        gate("prompt_perturbation_guard_clean", not any(issue["type"] == "multi_stream_perturbation_shape" for issue in hard_failures), "delimiter/reversal heuristics clean"),
        gate(
            "monitorability_coverage",
            monitorability_coverage >= float(policy.get("min_monitorability_coverage") or 0.95),
            f"coverage={monitorability_coverage:.3f}",
        ),
        gate(
            "verifier_score_floor",
            score >= float(policy.get("min_verifier_score") or 0.9),
            f"score={score:.3f}",
        ),
        gate("external_inference_zero", True, "deterministic local verification"),
    ]
    trigger_state = "GREEN" if all(item["passed"] for item in gates) else "RED"
    report = {
        "policy": "project_theseus_multi_stream_causal_verifier_v1",
        "created_utc": now(),
        "config": rel(resolve(args.policy)),
        "manifest": rel(resolve(args.manifest)),
        "trigger_state": trigger_state,
        "summary": {
            "case_count": len(cases),
            "monitorability_coverage": round(monitorability_coverage, 6),
            "verifier_score": round(score, 6),
            "hard_failure_count": len(hard_failures),
            "external_inference_calls": 0,
        },
        "case_results": case_results,
        "gates": gates,
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if trigger_state == "GREEN" else 1


def verify_case(case: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    streams = [str(item) for item in case.get("streams", []) if str(item)]
    required_streams = [str(item) for item in policy.get("streams", []) if str(item)]
    rows = case.get("stream_rows") if isinstance(case.get("stream_rows"), list) else []
    issues: list[dict[str, Any]] = []
    if streams != required_streams:
        issues.append(issue("stream_schema", "hard", f"streams={streams} expected={required_streams}"))
    if int(case.get("provenance", {}).get("copied_public_benchmark_item_chars") or 0) != 0:
        issues.append(issue("public_benchmark_copy", "hard", "copied public benchmark chars must be zero"))
    if case.get("scoring", {}).get("public_comparator_use") != "forbidden":
        issues.append(issue("public_comparator", "hard", str(case.get("scoring", {}).get("public_comparator_use"))))
    if len(rows) > int(get_path(policy, ["security", "max_rows_per_case"], 12)):
        issues.append(issue("runaway_reasoning_budget", "hard", f"rows={len(rows)}"))

    total_chars = 0
    total_tokens = 0
    stream_tokens = {stream: 0 for stream in required_streams}
    monitorability_present = False
    for row in rows:
        row_index = int(row.get("row_index") or 0)
        cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
        for stream in required_streams:
            if stream not in cells:
                issues.append(issue("stream_schema", "hard", f"row={row_index} missing={stream}"))
                continue
            cell = cells.get(stream) if isinstance(cells.get(stream), dict) else {}
            text = str(cell.get("text") or "")
            total_chars += len(text)
            if len(text) > int(get_path(policy, ["security", "max_cell_chars"], 1600)):
                issues.append(issue("cell_char_budget", "hard", f"row={row_index} stream={stream} chars={len(text)}"))
            if stream in {"critic_audit_stream", "residual_stream"} and text and text != "-":
                monitorability_present = True
            if perturbation_shape(text):
                issues.append(issue("multi_stream_perturbation_shape", "hard", f"row={row_index} stream={stream}"))
            token_count = int(cell.get("token_estimate") or estimate_tokens(text))
            total_tokens += token_count
            stream_tokens[stream] = stream_tokens.get(stream, 0) + token_count
            for dep in cell.get("depends_on", []) if isinstance(cell.get("depends_on"), list) else []:
                if not isinstance(dep, list) or len(dep) != 2:
                    issues.append(issue("dependency_shape", "hard", f"row={row_index} stream={stream} dep={dep}"))
                    continue
                dep_stream = str(dep[0])
                try:
                    dep_row = int(dep[1])
                except (TypeError, ValueError):
                    dep_row = row_index
                if dep_stream not in required_streams:
                    issues.append(issue("dependency_stream", "hard", f"row={row_index} stream={stream} dep_stream={dep_stream}"))
                if dep_row >= row_index:
                    issues.append(issue("future_dependency", "hard", f"row={row_index} stream={stream} dep={dep}"))
    if total_chars > int(get_path(policy, ["security", "max_total_chars_per_case"], 12000)):
        issues.append(issue("case_char_budget", "hard", f"chars={total_chars}"))
    critical_path_tokens = max(stream_tokens.values()) if stream_tokens else 0
    parallel_efficiency = 1.0 - (critical_path_tokens / total_tokens) if total_tokens else 0.0
    return {
        "case_id": case.get("case_id"),
        "task_id": case.get("task_id"),
        "passed": not any(row.get("severity") == "hard" for row in issues),
        "monitorability_present": monitorability_present,
        "total_non_idle_tokens": total_tokens,
        "critical_path_tokens": critical_path_tokens,
        "critical_path_fraction": round(critical_path_tokens / total_tokens, 6) if total_tokens else 1.0,
        "parallel_efficiency": round(max(0.0, parallel_efficiency), 6),
        "stream_tokens": stream_tokens,
        "issues": issues,
    }


def perturbation_shape(text: str) -> bool:
    if not text or text == "-":
        return False
    delimiter_hits = len(re.findall(r"[{}\[\]<>]", text))
    words = re.findall(r"[A-Za-z]{4,}", text)
    reversed_like = sum(1 for word in words if looks_reversed(word))
    if delimiter_hits >= 8 and delimiter_hits / max(1, len(text)) > 0.04:
        return True
    return bool(words and reversed_like / len(words) > 0.35 and len(words) >= 6)


def looks_reversed(word: str) -> bool:
    lower = word.lower()
    common = {"the", "and", "this", "that", "with", "from", "stream", "repair", "policy"}
    return lower[::-1] in common or bool(re.search(r"(ht|eht|dna|siht|maerts)", lower))


def verifier_score(case_results: list[dict[str, Any]], monitorability_coverage: float) -> float:
    if not case_results:
        return 0.0
    pass_rate = ratio(sum(1 for row in case_results if row.get("passed")), len(case_results))
    avg_efficiency = sum(float(row.get("parallel_efficiency") or 0.0) for row in case_results) / len(case_results)
    return max(0.0, min(1.0, (0.60 * pass_rate) + (0.25 * monitorability_coverage) + (0.15 * avg_efficiency)))


def estimate_tokens(text: str) -> int:
    if not text or text == "-":
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def issue(kind: str, severity: str, detail: str) -> dict[str, Any]:
    return {"type": kind, "severity": severity, "detail": detail}


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


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
