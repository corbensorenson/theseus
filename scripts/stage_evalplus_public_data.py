"""Stage small EvalPlus public calibration data on the runtime drive.

This is source setup, not training. The staged file contains benchmark
reference material for sandbox scoring only; Code LM public manifests still
export visible prompts without tests or canonical solutions.
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
DEFAULT_VERSION = "v0.1.10"
DEFAULT_DATASET = "HumanEvalPlus"
DEFAULT_ROOT = Path("D:/ProjectTheseus/resource_pantry/datasets/evalplus")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=["HumanEvalPlus"])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default="reports/stage_evalplus_public_data.json")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    gz_path = root / f"{args.dataset}-{args.version}.jsonl.gz"
    jsonl_path = root / f"{args.dataset}-{args.version}.jsonl"
    url = (
        f"https://github.com/evalplus/{args.dataset.lower()}_release/"
        f"releases/download/{args.version}/{args.dataset}.jsonl.gz"
    )

    downloaded = False
    if not gz_path.exists() and not jsonl_path.exists():
        with urllib.request.urlopen(url, timeout=120) as response, gz_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        downloaded = True

    decompressed = False
    if not jsonl_path.exists() and gz_path.exists():
        with gzip.open(gz_path, "rt", encoding="utf-8") as src, jsonl_path.open("w", encoding="utf-8") as dst:
            shutil.copyfileobj(src, dst)
        decompressed = True

    row_count, sample = inspect_jsonl(jsonl_path)
    payload = {
        "policy": "project_theseus_stage_evalplus_public_data_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if row_count >= 32 else "YELLOW",
        "dataset": args.dataset,
        "version": args.version,
        "url": url,
        "root": str(root).replace("\\", "/"),
        "jsonl": str(jsonl_path).replace("\\", "/"),
        "jsonl_gz": str(gz_path).replace("\\", "/"),
        "downloaded": downloaded,
        "decompressed": decompressed,
        "row_count": row_count,
        "sample_keys": sorted(sample.keys()) if isinstance(sample, dict) else [],
        "public_benchmark_solutions_included": True,
        "solution_use_policy": "scorer_only_not_exported_to_student_or_training",
        "training_use_allowed": False,
        "public_tests_exported_to_generator": False,
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] == "GREEN" else 2


def inspect_jsonl(path: Path) -> tuple[int, dict[str, Any]]:
    if not path.exists():
        return 0, {}
    count = 0
    sample: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            if not sample:
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    raw = {}
                sample = raw if isinstance(raw, dict) else {}
    return count, sample


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
