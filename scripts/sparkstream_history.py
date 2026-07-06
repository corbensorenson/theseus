"""Append-only SparkStream metric history.

The dashboard reads the compact JSON report produced here to draw improvement
curves over days or weeks. The JSONL file is the durable source of truth.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
JSONL = REPORTS / "sparkstream_metrics.jsonl"
DEFAULT_OUT = REPORTS / "sparkstream_history.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--max-points", type=int, default=2000)
    args = parser.parse_args()

    if args.append:
        append_jsonl(JSONL, current_point())
    history = build_history(args.max_points)
    write_json(ROOT / args.out, history)
    print(json.dumps(history, indent=2))
    return 0


def current_point() -> dict[str, Any]:
    benchmarks = read_json(REPORTS / "benchmark_ledger.json")
    candidate = read_json(REPORTS / "candidate_promotion_gate.json")
    preflight = read_json(REPORTS / "training_preflight_report.json")
    checkpoints = read_json(REPORTS / "checkpoint_registry.json")
    data = read_json(REPORTS / "training_data_inventory.json")
    rl = read_json(REPORTS / "rl_benchmark_registry.json")
    status = read_json(REPORTS / "sparkstream_status.json")
    return {
        "created_utc": now(),
        "status_phase": status.get("phase"),
        "profile": status.get("profile"),
        "benchmarks": benchmark_points(benchmarks),
        "candidate": {
            "promote": candidate.get("promote"),
            "passed": candidate.get("passed"),
            "total": candidate.get("total"),
            "public_accuracy": get_path(candidate, ["scores", "public_accuracy"], None),
            "seed49_regression_accuracy": get_path(candidate, ["scores", "seed49_regression_accuracy"], None),
            "seed55_frontier_accuracy": get_path(candidate, ["scores", "seed55_frontier_accuracy"], None),
        },
        "preflight": {
            "heavy_training_allowed": preflight.get("heavy_training_allowed"),
            "passed": preflight.get("passed"),
            "total": preflight.get("total"),
            "blocker_count": preflight.get("blocker_count"),
            "warning_count": preflight.get("warning_count"),
        },
        "checkpoints": {
            "count": len(checkpoints.get("checkpoints", [])) if isinstance(checkpoints, dict) else 0,
            "latest": latest_checkpoint(checkpoints),
        },
        "data": get_path(data, ["summary"], {}),
        "rl": get_path(rl, ["summary"], {}),
    }


def benchmark_points(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "name": item.get("benchmark_name"),
                "score": item.get("score"),
                "residual": item.get("residual"),
                "lifecycle": item.get("lifecycle"),
                "wall_type": item.get("wall_type"),
                "threshold": get_path(item, ["graduation_policy", "current_threshold"], None),
                "floor": get_path(item, ["graduation_policy", "floor_threshold"], None),
            }
        )
    return rows


def build_history(max_points: int) -> dict[str, Any]:
    rows = read_jsonl_tail(JSONL, max_points)
    benchmark_series: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        created = row.get("created_utc")
        for bench in row.get("benchmarks", []):
            name = bench.get("name")
            if not name:
                continue
            benchmark_series.setdefault(str(name), []).append(
                {
                    "created_utc": created,
                    "score": bench.get("score"),
                    "residual": bench.get("residual"),
                    "threshold": bench.get("threshold"),
                    "lifecycle": bench.get("lifecycle"),
                }
            )
    return {
        "policy": "sparkstream_history_v0",
        "updated_utc": now(),
        "points": rows,
        "benchmark_series": benchmark_series,
        "summary": {
            "points": len(rows),
            "benchmarks_tracked": len(benchmark_series),
            "first_utc": rows[0].get("created_utc") if rows else None,
            "last_utc": rows[-1].get("created_utc") if rows else None,
        },
    }


def latest_checkpoint(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    rows = value.get("checkpoints") or []
    return rows[-1] if rows else {}


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
