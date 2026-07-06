"""Export the local BabyLM BLIMP filtered Hugging Face cache to JSONL.

This script performs local data preparation only. It does not call hosted model
APIs or run any external inference.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


def read_arrow_table(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        try:
            return ipc.open_stream(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            return ipc.open_file(source).read_all()


def row_value(table: pa.Table, name: str, idx: int) -> object:
    if name not in table.column_names:
        return None
    return table[name][idx].as_py()


def iter_rows(cache_root: Path):
    for arrow_path in sorted(cache_root.rglob("baby_lm-blimp-filtered-train.arrow")):
        config = arrow_path.parents[3].name
        table = read_arrow_table(arrow_path)
        for idx in range(table.num_rows):
            good = row_value(table, "sentence_good", idx)
            bad = row_value(table, "sentence_bad", idx)
            if not good or not bad:
                continue
            uid = row_value(table, "UID", idx) or config
            field = row_value(table, "field", idx) or config
            term = row_value(table, "linguistics_term", idx) or ""
            pair_id = row_value(table, "pair_id", idx)
            yield {
                "sentence_good": str(good),
                "sentence_bad": str(bad),
                "rule": str(uid),
                "field": str(field),
                "linguistics_term": str(term),
                "pair_id": pair_id,
                "source": "local_babylm_blimp_filtered_cache",
            }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-eval", required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    rows = list(iter_rows(cache_root))
    if args.limit > 0:
        rows = rows[: args.limit]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    eval_count = max(1, int(len(rows) * args.eval_fraction))
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]

    write_jsonl(Path(args.out_train), train_rows)
    write_jsonl(Path(args.out_eval), eval_rows)
    print(
        json.dumps(
            {
                "cache_root": str(cache_root),
                "rows": len(rows),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "out_train": args.out_train,
                "out_eval": args.out_eval,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
