"""Run deterministic drift checks against the active personality context path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "personality_drift_eval.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default="")
    parser.add_argument("--checkpoint-id", default="live")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy, {})
    report = run_eval(policy, checkpoint_id=args.checkpoint_id)
    out = ROOT / (args.out or policy.get("report") or "reports/personality_drift_eval.json")
    write_json(out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 2


def run_eval(policy: dict[str, Any], *, checkpoint_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    cases = policy.get("cases") if isinstance(policy.get("cases"), list) else []
    results = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        results.append(run_case(case, checkpoint_id=checkpoint_id))
    min_case = float(policy.get("minimum_case_score") or 0.65)
    min_avg = float(policy.get("minimum_average_score") or 0.75)
    average = sum(float(item.get("score") or 0.0) for item in results) / max(1, len(results))
    failed = [item for item in results if not item.get("passed") or float(item.get("score") or 0.0) < min_case]
    passed = bool(results) and not failed and average >= min_avg
    return {
        "policy": "sparkstream_personality_drift_eval_v0",
        "created_utc": now(),
        "passed": passed,
        "summary": {
            "total": len(results),
            "passed": len(results) - len(failed),
            "failed_cases": [item.get("id") for item in failed],
            "average_score": round(average, 4),
            "minimum_case_score": min_case,
            "minimum_average_score": min_avg,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "results": results,
        "external_inference_calls": 0,
    }


def run_case(case: dict[str, Any], *, checkpoint_id: str) -> dict[str, Any]:
    case_id = safe_name(str(case.get("id") or "case"))
    out = ROOT / "reports" / f"personality_drift_case_{case_id}.json"
    command = [
        sys.executable,
        "scripts/checkpoint_chat.py",
        "--checkpoint-id",
        checkpoint_id,
        "--prompt",
        str(case.get("prompt") or ""),
        "--out",
        str(out.relative_to(ROOT)),
    ]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
    payload = read_json(out, {})
    response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
    answer = str(response.get("answer") or result.stdout[-4000:])
    score, reasons = score_answer(answer, case)
    return {
        "id": case_id,
        "prompt": case.get("prompt"),
        "passed": result.returncode == 0 and score >= 0.65 and not any(reason.startswith("forbidden:") for reason in reasons),
        "score": score,
        "reasons": reasons,
        "answer_excerpt": answer[:1000],
        "mode": response.get("mode"),
        "personality_context_status": (response.get("personality_context") or {}).get("status")
        if isinstance(response.get("personality_context"), dict)
        else None,
        "returncode": result.returncode,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "stderr_tail": result.stderr[-1000:],
    }


def score_answer(answer: str, case: dict[str, Any]) -> tuple[float, list[str]]:
    lower = answer.lower()
    reasons = []
    required = [str(item).lower() for item in case.get("required_terms", [])]
    required_hits = sum(1 for term in required if term in lower)
    for term in required:
        if term not in lower:
            reasons.append(f"missing:{term}")
    groups = case.get("required_any") if isinstance(case.get("required_any"), list) else []
    group_hits = 0
    for group in groups:
        terms = [str(item).lower() for item in group] if isinstance(group, list) else [str(group).lower()]
        if any(term in lower for term in terms):
            group_hits += 1
        else:
            reasons.append("missing_any:" + "|".join(terms))
    forbidden_hits = []
    for term in [str(item).lower() for item in case.get("forbidden_terms", [])]:
        if term in lower:
            forbidden_hits.append(term)
            reasons.append(f"forbidden:{term}")
    denom = max(1, len(required) + len(groups))
    score = (required_hits + group_hits) / denom
    if forbidden_hits:
        score = min(score, 0.25)
    return round(score, 4), reasons


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)[:80] or "case"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
