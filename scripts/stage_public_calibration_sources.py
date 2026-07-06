#!/usr/bin/env python3
"""Stage licensed scorer-only public calibration sources locally.

The staged files are for bounded public calibration scoring only. They are not
private training rows and should not be committed to git.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "resource_pantry" / "git"
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default="reports/public_calibration_source_staging.json")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional export cap per dataset; 0 writes all loaded rows.")
    args = parser.parse_args()

    root = resolve(args.root)
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "policy": "project_theseus_public_calibration_source_staging_v1",
        "created_utc": now(),
        "root": rel_or_abs(root),
        "sources": [],
        "rules": {
            "use": "bounded public calibration scoring only",
            "private_training": "forbidden",
            "public_solutions_or_tests_to_generator": "forbidden",
            "commit_payloads_to_git": "forbidden",
        },
        "external_inference_calls": 0,
    }

    report["sources"].append(stage_mbpp(root, max_rows=args.max_rows))
    report["sources"].append(stage_evalplus(root, max_rows=args.max_rows))
    report["sources"].append(stage_bigcodebench(root, max_rows=args.max_rows))
    report["sources"].append(stage_humaneval(root))
    report["summary"] = {
        "source_count": len(report["sources"]),
        "ready_count": sum(1 for row in report["sources"] if row.get("ready")),
        "all_ready": all(bool(row.get("ready")) for row in report["sources"]),
        "rows": {str(row.get("source_id")): int(row.get("row_count") or 0) for row in report["sources"]},
    }
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if report["summary"]["all_ready"] else 2


def stage_mbpp(root: Path, *, max_rows: int) -> dict[str, Any]:
    target_dir = root / "mbpp"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "sanitized-mbpp.json"
    row = base_row("source_mbpp", "mbpp", "apache-2.0", target)
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("mbpp", "sanitized", split="test")
        rows = [dict(item) for item in ds]
        if max_rows > 0:
            rows = rows[:max_rows]
        target.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        row.update({"ready": True, "row_count": len(rows), "dataset": "mbpp/sanitized:test"})
    except Exception as exc:  # pragma: no cover - environment-dependent
        row.update({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
    return row


def stage_evalplus(root: Path, *, max_rows: int) -> dict[str, Any]:
    target_dir = root / "evalplus"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "HumanEvalPlus-v0.1.10.jsonl"
    row = base_row("source_evalplus", "evalplus", "apache-2.0", target)
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("evalplus/humanevalplus", split="test")
        rows = [dict(item) for item in ds]
        if max_rows > 0:
            rows = rows[:max_rows]
        write_jsonl(target, rows)
        row.update({"ready": True, "row_count": len(rows), "dataset": "evalplus/humanevalplus:test"})
    except Exception as exc:  # pragma: no cover - environment-dependent
        row.update({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
    return row


def stage_bigcodebench(root: Path, *, max_rows: int) -> dict[str, Any]:
    target_dir = root / "bigcodebench"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "BigCodeBench-v0.1.4.jsonl"
    row = base_row("source_bigcodebench", "bigcodebench", "apache-2.0", target)
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
        rows = [dict(item) for item in ds]
        if max_rows > 0:
            rows = rows[:max_rows]
        write_jsonl(target, rows)
        row.update({"ready": True, "row_count": len(rows), "dataset": "bigcode/bigcodebench:v0.1.4"})
    except Exception as exc:  # pragma: no cover - environment-dependent
        row.update({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
    return row


def stage_humaneval(root: Path) -> dict[str, Any]:
    target_dir = root / "human_eval" / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "HumanEval.jsonl.gz"
    row = base_row("source_human_eval", "human_eval", "mit", target)
    try:
        with urllib.request.urlopen(HUMANEVAL_URL, timeout=60) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        count = 0
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        row.update({"ready": count > 0, "row_count": count, "dataset": HUMANEVAL_URL})
    except Exception as exc:  # pragma: no cover - environment-dependent
        row.update({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
    return row


def base_row(card_id: str, source_id: str, license_spdx: str, path: Path) -> dict[str, Any]:
    return {
        "card_id": card_id,
        "source_id": source_id,
        "license_spdx": license_spdx,
        "path": rel_or_abs(path),
        "ready": False,
        "score_semantics": "public scorer-only calibration payload",
        "private_training_allowed": False,
    }


def resolve(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
