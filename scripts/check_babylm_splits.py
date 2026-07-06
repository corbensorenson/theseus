"""Check BabyLM/BLIMP JSONL splits for exact pair and sentence leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Named split in name=path form. May be repeated.",
    )
    parser.add_argument("--out", default="reports/babylm_split_leakage_report.json")
    parser.add_argument(
        "--strict-bridge",
        action="store_true",
        help="Treat exact bridge overlap as a blocking leakage failure.",
    )
    args = parser.parse_args()

    splits = parse_splits(args.split)
    if not splits:
        splits = {
            "train": Path("data/babylm_blimp_filtered_train.jsonl"),
            "public_eval": Path("data/babylm_blimp_filtered_eval.jsonl"),
            "mutated_seed49": Path("data/babylm_mutated_holdout_seed49.jsonl"),
            "mutated_seed55": Path("data/babylm_mutated_holdout_seed55.jsonl"),
            "wh_bridge": Path("benchmarks/bridges/babylm_wh_gap_bridge.jsonl"),
        }

    inventories = {name: inventory(path) for name, path in splits.items()}
    pair_overlaps = []
    sentence_overlaps = []
    names = sorted(inventories)
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            left_inv = inventories[left]
            right_inv = inventories[right]
            pair_overlap = sorted(left_inv["pair_hashes"] & right_inv["pair_hashes"])
            sentence_overlap = sorted(
                left_inv["sentence_hashes"] & right_inv["sentence_hashes"]
            )
            pair_overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "count": len(pair_overlap),
                    "examples": pair_overlap[:10],
                }
            )
            sentence_overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "count": len(sentence_overlap),
                    "examples": sentence_overlap[:10],
                }
            )

    blocking_pair_overlaps = [
        item
        for item in pair_overlaps
        if item["count"] > 0
        and (args.strict_bridge or ("bridge" not in item["left"] and "bridge" not in item["right"]))
    ]
    total_pair_overlaps = sum(item["count"] for item in pair_overlaps)
    total_blocking_pair_overlaps = sum(item["count"] for item in blocking_pair_overlaps)
    total_sentence_overlaps = sum(item["count"] for item in sentence_overlaps)
    split_classification = classify_splits(names, pair_overlaps, sentence_overlaps)
    missing = [name for name, inv in inventories.items() if not inv["exists"]]
    report = {
        "policy": "local_only_no_external_inference",
        "methodology": "babylm_exact_leakage_check",
        "ok": not missing and total_blocking_pair_overlaps == 0,
        "strict_ok": not missing and total_pair_overlaps == 0 and total_sentence_overlaps == 0,
        "missing_splits": missing,
        "splits": {
            name: {
                "path": str(splits[name]),
                "exists": inv["exists"],
                "rows": inv["rows"],
                "unique_pairs": len(inv["pair_hashes"]),
                "unique_sentences": len(inv["sentence_hashes"]),
                "by_rule": dict(sorted(inv["by_rule"].items())),
            }
            for name, inv in inventories.items()
        },
        "pair_overlaps": pair_overlaps,
        "blocking_pair_overlaps": blocking_pair_overlaps,
        "sentence_overlaps": sentence_overlaps,
        "total_pair_overlaps": total_pair_overlaps,
        "total_blocking_pair_overlaps": total_blocking_pair_overlaps,
        "total_sentence_overlaps": total_sentence_overlaps,
        "split_classification": split_classification,
        "recommendation": recommendation(
            missing,
            total_blocking_pair_overlaps,
            total_pair_overlaps,
            total_sentence_overlaps,
            split_classification,
        ),
    }
    write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def parse_splits(values: list[str]) -> dict[str, Path]:
    splits: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"split must be name=path, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"split name cannot be empty in {value!r}")
        splits[name] = Path(path.strip())
    return splits


def inventory(path: Path) -> dict[str, Any]:
    out = {
        "exists": path.exists(),
        "rows": 0,
        "pair_hashes": set(),
        "sentence_hashes": set(),
        "by_rule": defaultdict(int),
    }
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            good = normalize(row.get("sentence_good") or row.get("good") or "")
            bad = normalize(row.get("sentence_bad") or row.get("bad") or "")
            if not good or not bad:
                continue
            out["rows"] += 1
            out["pair_hashes"].add(stable_hash(f"{good}\n{bad}"))
            out["sentence_hashes"].add(stable_hash(good))
            out["sentence_hashes"].add(stable_hash(bad))
            rule = str(row.get("rule") or row.get("UID") or "unknown")
            out["by_rule"][rule] += 1
    return out


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def recommendation(
    missing: list[str],
    blocking_pair_overlaps: int,
    pair_overlaps: int,
    sentence_overlaps: int,
    split_classification: dict[str, Any],
) -> dict[str, Any]:
    actions = []
    if missing:
        actions.append("generate_missing_splits")
    if blocking_pair_overlaps:
        actions.append("regenerate_or_filter_exact_pair_leaks")
    elif pair_overlaps:
        actions.append("review_bridge_overlap_if_used_as_private_holdout")
    if sentence_overlaps:
        actions.append("review_sentence_reuse_before_candidate_training")
    for name, classification in split_classification.items():
        if classification["quality"] == "mutated_frontier_not_pristine_private_holdout":
            actions.append(f"classify_{name}_as_mutated_frontier")
    if not actions:
        actions.append("splits_clear_for_training_gate")
    return {
        "actions": actions,
        "candidate_training_allowed": not missing and blocking_pair_overlaps == 0,
        "strict_private_holdout_quality": not missing
        and pair_overlaps == 0
        and sentence_overlaps == 0,
    }


def classify_splits(
    names: list[str],
    pair_overlaps: list[dict[str, Any]],
    sentence_overlaps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify holdout quality without pretending mutated fronts are pristine."""

    out: dict[str, Any] = {}
    pair_counts = overlap_counts_by_split(pair_overlaps)
    sentence_counts = overlap_counts_by_split(sentence_overlaps)
    for name in names:
        pairs = pair_counts.get(name, 0)
        sentences = sentence_counts.get(name, 0)
        if pairs == 0 and sentences == 0:
            quality = "strict_private_holdout_candidate"
        elif pairs == 0 and "mutated" in name:
            quality = "mutated_frontier_not_pristine_private_holdout"
        elif pairs == 0:
            quality = "calibration_or_curriculum_split_with_sentence_reuse"
        else:
            quality = "exact_pair_overlap_review_required"
        out[name] = {
            "quality": quality,
            "exact_pair_overlaps": pairs,
            "exact_sentence_overlaps": sentences,
            "private_holdout_claim_allowed": pairs == 0 and sentences == 0,
            "frontier_claim_allowed": pairs == 0,
        }
    return out


def overlap_counts_by_split(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        count = int(item.get("count") or 0)
        counts[str(item.get("left"))] += count
        counts[str(item.get("right"))] += count
    return dict(counts)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
