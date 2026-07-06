"""Stage governed public code benchmark payloads on D: for evaluation only.

This script downloads public benchmark task data into the resource pantry so
the broad transfer matrix can evaluate real task adapters instead of
loader-only manifests. These payloads are public calibration data only: they
must not be admitted into private training or residual repair datasets.
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
PANTRY = Path("D:/ProjectTheseus/resource_pantry/datasets")

BIGCODEBENCH_URL = (
    "https://huggingface.co/datasets/bigcode/bigcodebench/resolve/main/data/"
    "v0.1.4-00000-of-00001.parquet"
)
LIVECODEBENCH_RELEASE_V1_SHARDS = [
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/a16d03780493b939b3601fb9da2ac3ed2b23caa2/release_v1/test-00000-of-00003.parquet",
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/a16d03780493b939b3601fb9da2ac3ed2b23caa2/release_v1/test-00001-of-00003.parquet",
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/a16d03780493b939b3601fb9da2ac3ed2b23caa2/release_v1/test-00002-of-00003.parquet",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/public_code_benchmark_data_stage.json")
    parser.add_argument("--skip-bigcodebench", action="store_true")
    parser.add_argument("--skip-livecodebench", action="store_true")
    parser.add_argument(
        "--live-shards",
        type=int,
        default=3,
        help="Number of LiveCodeBench release_v1 parquet shards to stage. Use 1 for a fast adapter smoke.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    if not args.skip_bigcodebench:
        rows.append(stage_bigcodebench())
    if not args.skip_livecodebench:
        for index, url in enumerate(LIVECODEBENCH_RELEASE_V1_SHARDS[: max(0, args.live_shards)]):
            rows.append(stage_livecodebench_shard(index, url))

    report = {
        "policy": "project_theseus_public_code_benchmark_data_stage_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row.get("available") for row in rows) else "YELLOW",
        "storage_root": str(PANTRY).replace("\\", "/"),
        "rows": rows,
        "training_admission": {
            "public_benchmark_solutions_or_tests_may_train": False,
            "use": "public calibration/evaluation only",
            "score_semantics": "dataset staging is operational readiness, not student learning evidence",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 1


def stage_bigcodebench() -> dict[str, Any]:
    root = PANTRY / "bigcodebench"
    parquet_path = root / "v0.1.4-00000-of-00001.parquet"
    jsonl_path = root / "BigCodeBench-v0.1.4.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    error = ""
    try:
        downloaded = download_if_missing(BIGCODEBENCH_URL, parquet_path)
    except Exception as exc:
        downloaded = False
        error = f"{exc.__class__.__name__}: {exc}"
    converted = False
    if parquet_path.exists() and not jsonl_path.exists():
        try:
            converted = parquet_to_jsonl(parquet_path, jsonl_path)
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
    return {
        "dataset": "bigcodebench",
        "available": jsonl_path.exists(),
        "path": str(jsonl_path).replace("\\", "/"),
        "parquet_path": str(parquet_path).replace("\\", "/"),
        "downloaded": downloaded,
        "converted": converted,
        "error": error,
        "source_url": BIGCODEBENCH_URL,
        "license": "apache-2.0",
        "benchmark_evidence_level": "public_calibration_eval_only_no_training",
    }


def stage_livecodebench_shard(index: int, url: str) -> dict[str, Any]:
    root = PANTRY / "livecodebench" / "release_v1"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"test-{index:05d}-of-00003.parquet"
    error = ""
    try:
        downloaded = download_if_missing(url, path)
    except Exception as exc:
        downloaded = False
        error = f"{exc.__class__.__name__}: {exc}"
    return {
        "dataset": "livecodebench_code_generation_lite",
        "available": path.exists(),
        "path": str(path).replace("\\", "/"),
        "downloaded": downloaded,
        "error": error,
        "source_url": url,
        "license": "cc",
        "benchmark_evidence_level": "public_calibration_eval_only_no_training",
    }


def download_if_missing(url: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return False
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(path)
    return True


def parquet_to_jsonl(parquet_path: Path, jsonl_path: Path) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False
    rows = pd.read_parquet(parquet_path).to_dict(orient="records")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
