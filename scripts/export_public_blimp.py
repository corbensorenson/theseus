"""Export the fetched public BLIMP repo into balanced local JSONL splits.

This is local benchmark preparation only. It reads official BLIMP JSONL files
already fetched under data/public_benchmarks and writes train/eval splits that
the SymLiquid BabyLM probe trainer can consume directly.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def iter_blimp_rows(data_dir: Path):
    for path in sorted(data_dir.glob("*.jsonl")):
        uid_from_file = path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("sentence_good") or not row.get("sentence_bad"):
                    continue
                row.setdefault("UID", uid_from_file)
                row.setdefault("rule", row.get("UID", uid_from_file))
                row["source"] = "public_blimp"
                row["source_file"] = path.name
                yield row


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/public_benchmarks/blimp/data")
    parser.add_argument("--out-train", default="data/public_blimp_train.jsonl")
    parser.add_argument("--out-eval", default="data/public_blimp_eval.jsonl")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-per-rule", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows_by_rule: dict[str, list[dict]] = defaultdict(list)
    for row in iter_blimp_rows(Path(args.data_dir)):
        rule = str(row.get("UID") or row.get("rule") or "unknown")
        rows_by_rule[rule].append(row)

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for rule, rows in sorted(rows_by_rule.items()):
        rng.shuffle(rows)
        if args.limit_per_rule > 0:
            rows = rows[: args.limit_per_rule]
        eval_count = max(1, int(len(rows) * args.eval_fraction))
        eval_rows.extend(rows[:eval_count])
        train_rows.extend(rows[eval_count:])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    write_jsonl(Path(args.out_train), train_rows)
    write_jsonl(Path(args.out_eval), eval_rows)
    print(
        json.dumps(
            {
                "rules": len(rows_by_rule),
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
