"""Governed synthetic-data curation for SparkStream/RMI.

The first production path is deliberately conservative: generate local,
residual-targeted BabyLM minimal pairs from templates and feature-preserving
mutations, reject anything that overlaps eval/holdout sentences, score quality,
cap synthetic ratio, and write a blended training file for the ratchet runner.

No external model calls are made here. Teacher/web-based synthetic generation
stays proposal-only until separate provenance, license, and approval gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_babylm_mutated_holdout as babylm_factory  # noqa: E402


DEFAULT_POLICY = ROOT / "configs" / "synthetic_data_policy.json"
DEFAULT_OUT = ROOT / "reports" / "synthetic_data_curator.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--residual-escrow", default="reports/residual_escrow.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--out-data", default="")
    parser.add_argument("--blend-out", default="")
    parser.add_argument("--dataset-card-out", default="")
    parser.add_argument("--external-pairs-report", default="")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blend-total", type=int, default=0)
    parser.add_argument("--max-synthetic-ratio", type=float, default=-1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    babylm_policy = policy.get("babylm") or {}
    gates = policy.get("quality_gates") or {}
    collapse_guard = policy.get("collapse_guard") or {}
    seed = args.seed or int(babylm_policy.get("seed", 2055))
    count = args.count or int(babylm_policy.get("count", 6400))
    blend_total = args.blend_total or int(babylm_policy.get("blend_total", 50000))
    max_synthetic_ratio = (
        args.max_synthetic_ratio
        if args.max_synthetic_ratio >= 0
        else float(collapse_guard.get("max_synthetic_ratio", 0.12))
    )
    max_external_pair_ratio = float(collapse_guard.get("max_external_pair_ratio", 0.02))
    out_data = ROOT / (args.out_data or babylm_policy.get("synthetic_out", "data/synthetic/babylm_residual_targeted_current.jsonl"))
    blend_out = ROOT / (args.blend_out or babylm_policy.get("blend_out", "data/synthetic/babylm_train_plus_synthetic_current.jsonl"))
    dataset_card_out = ROOT / (
        args.dataset_card_out
        or babylm_policy.get("dataset_card_out", "data/synthetic/babylm_residual_targeted_current.dataset_card.json")
    )
    sources = [ROOT / path for path in babylm_policy.get("sources", [])]
    excludes = [ROOT / path for path in babylm_policy.get("excludes", [])]
    external_pairs_report = ROOT / (
        args.external_pairs_report
        or babylm_policy.get("external_pairs_report", "reports/training_data_sampler.json")
    )

    residual = read_json(ROOT / args.residual_escrow)
    targets = select_targets(residual, int(babylm_policy.get("max_target_rules", 18)))
    source_rows = babylm_factory.read_source_rows(sources)
    exclude_rows = babylm_factory.read_source_rows(excludes)
    candidate_rows = generate_candidates(
        source_rows=source_rows,
        exclude_rows=exclude_rows,
        targets=targets,
        count=max(count * 2, count + 2000),
        seed=seed,
    )
    accepted, rejected = curate_candidates(
        candidate_rows,
        source_rows=source_rows,
        exclude_rows=exclude_rows,
        targets=targets,
        min_quality=float(gates.get("min_quality_score", 0.7)),
        max_rule_share=float(gates.get("max_rule_share", 0.18)),
    )
    accepted = accepted[:count]
    external_rows, external_summary = load_external_pairwise_rows(
        external_pairs_report,
        source_rows=source_rows,
        exclude_rows=exclude_rows,
    )
    blend_rows = build_blend(
        source_rows=source_rows,
        synthetic_rows=accepted,
        external_rows=external_rows,
        total=blend_total,
        max_synthetic_ratio=max_synthetic_ratio,
        max_external_pair_ratio=max_external_pair_ratio,
        seed=seed,
    )
    verification = verify_dataset(
        synthetic_rows=accepted,
        external_rows=external_rows,
        blend_rows=blend_rows,
        source_rows=source_rows,
        exclude_rows=exclude_rows,
        targets=targets,
        gates=gates,
        max_synthetic_ratio=max_synthetic_ratio,
        max_external_pair_ratio=max_external_pair_ratio,
        external_summary=external_summary,
    )
    report = build_report(
        policy=policy,
        residual=residual,
        targets=targets,
        source_rows=source_rows,
        exclude_rows=exclude_rows,
        accepted=accepted,
        rejected=rejected,
        external_rows=external_rows,
        external_summary=external_summary,
        blend_rows=blend_rows,
        verification=verification,
        out_data=out_data,
        blend_out=blend_out,
        dataset_card_out=dataset_card_out,
        dry_run=args.dry_run,
        seed=seed,
    )
    dataset_card = build_dataset_card(report, policy)

    if not args.dry_run:
        write_jsonl(out_data, accepted)
        write_jsonl(blend_out, blend_rows)
        write_json(dataset_card_out, dataset_card)
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["training_ready"] else 1


def select_targets(residual: dict[str, Any], max_targets: int) -> dict[str, list[str]]:
    targets = residual.get("active_diagnostic_targets") or []
    rules: list[str] = []
    terms: list[str] = []
    for item in targets:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        if item.get("kind") == "term":
            append_unique(terms, name)
        else:
            append_unique(rules, name)
        if len(rules) + len(terms) >= max_targets:
            break
    return {"rules": rules, "terms": terms}


def generate_candidates(
    *,
    source_rows: list[dict[str, Any]],
    exclude_rows: list[dict[str, Any]],
    targets: dict[str, list[str]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    target_rules = targets.get("rules") or []
    target_terms = targets.get("terms") or []
    rows: list[dict[str, Any]] = []
    generated_pairs: set[str] = set()
    generated_sentences: set[str] = set()
    all_existing = source_rows + exclude_rows
    existing_pairs = {pair_key(row) for row in all_existing}
    existing_sentences = {sentence_key(sentence) for row in all_existing for sentence in row_sentences(row)}

    template_rows = babylm_factory.generate_template_rows(rng, count * 2, target_rules, target_terms)
    target_source_rows = babylm_factory.filter_target_rows(source_rows, target_rules, target_terms)
    mutation_budget = count * 3
    mutations: list[dict[str, Any]] = []
    for _ in range(mutation_budget):
        if not target_source_rows:
            break
        mutated = babylm_factory.mutate_source_row(rng.choice(target_source_rows), rng)
        if mutated is not None:
            mutations.append(mutated)

    for raw in interleave(template_rows, mutations):
        if len(rows) >= count:
            break
        candidate = dict(raw)
        candidate["source"] = "symliquid_synthetic_data_curator"
        candidate["generation_policy"] = "local_only_no_external_inference"
        candidate["generation_strategy"] = candidate.get("mutation_kind", "template")
        candidate["targeted_by_residual"] = is_targeted(candidate, targets)
        good = clean_sentence(candidate.get("sentence_good", ""))
        bad = clean_sentence(candidate.get("sentence_bad", ""))
        candidate["sentence_good"] = good
        candidate["sentence_bad"] = bad
        key = pair_key(candidate)
        sent_keys = {sentence_key(good), sentence_key(bad)}
        if not good or not bad or good == bad:
            continue
        if key in existing_pairs or key in generated_pairs:
            continue
        if sent_keys & existing_sentences or sent_keys & generated_sentences:
            continue
        generated_pairs.add(key)
        generated_sentences |= sent_keys
        rows.append(candidate)
    return rows


def curate_candidates(
    rows: list[dict[str, Any]],
    *,
    source_rows: list[dict[str, Any]],
    exclude_rows: list[dict[str, Any]],
    targets: dict[str, list[str]],
    min_quality: float,
    max_rule_share: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exclude_pairs = {pair_key(row) for row in source_rows + exclude_rows}
    exclude_sentences = {sentence_key(sentence) for row in source_rows + exclude_rows for sentence in row_sentences(row)}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rule_counts: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        scored = dict(row)
        quality = score_row(scored, targets, exclude_pairs, exclude_sentences)
        scored["quality"] = quality
        rule = str(scored.get("rule") or "unknown")
        projected_share = (rule_counts[rule] + 1) / max(1, len(accepted) + 1)
        if quality["score"] < min_quality:
            scored["rejection_reason"] = "quality_below_threshold"
            rejected.append(scored)
            continue
        if projected_share > max_rule_share and len(accepted) > 20:
            scored["rejection_reason"] = "rule_share_cap"
            rejected.append(scored)
            continue
        scored["synthetic_id"] = f"synthetic_babylm_{idx:06d}"
        scored["synthetic_created_utc"] = now()
        accepted.append(scored)
        rule_counts[rule] += 1
    return accepted, rejected


def score_row(
    row: dict[str, Any],
    targets: dict[str, list[str]],
    exclude_pairs: set[str],
    exclude_sentences: set[str],
) -> dict[str, Any]:
    reasons: list[str] = []
    score = 0.45
    good = clean_sentence(row.get("sentence_good", ""))
    bad = clean_sentence(row.get("sentence_bad", ""))
    if good and bad and good != bad:
        score += 0.15
        reasons.append("valid_minimal_pair")
    if 3 <= len(good.split()) <= 36 and 3 <= len(bad.split()) <= 36:
        score += 0.08
        reasons.append("bounded_sentence_length")
    if abs(len(good.split()) - len(bad.split())) <= 8:
        score += 0.05
        reasons.append("pair_lengths_comparable")
    if is_targeted(row, targets):
        score += 0.18
        reasons.append("targets_active_residual")
    if pair_key(row) not in exclude_pairs:
        score += 0.06
        reasons.append("no_pair_overlap")
    if sentence_key(good) not in exclude_sentences and sentence_key(bad) not in exclude_sentences:
        score += 0.08
        reasons.append("no_sentence_overlap")
    if row.get("source") == "symliquid_synthetic_data_curator":
        score += 0.04
        reasons.append("local_provenance")
    if row.get("generation_policy") == "local_only_no_external_inference":
        score += 0.04
        reasons.append("no_external_inference")
    return {"score": round(min(score, 1.0), 4), "reasons": reasons}


def build_blend(
    *,
    source_rows: list[dict[str, Any]],
    synthetic_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    total: int,
    max_synthetic_ratio: float,
    max_external_pair_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    if total <= 0:
        return []
    rng = random.Random(seed + 991)
    synthetic_cap = min(len(synthetic_rows), max(0, int(total * max_synthetic_ratio)))
    external_cap = min(len(external_rows), max(0, int(total * max_external_pair_ratio)))
    real_cap = max(0, total - synthetic_cap - external_cap)
    real_rows = [dict(row) for row in source_rows[:real_cap]]
    synth_rows = [dict(row) for row in synthetic_rows[:synthetic_cap]]
    ext_rows = [dict(row) for row in external_rows[:external_cap]]
    for row in real_rows:
        row.setdefault("training_origin", "real_seed_data")
    for row in synth_rows:
        row["training_origin"] = "synthetic_verified_residual_targeted"
    for row in ext_rows:
        row["training_origin"] = "external_open_sample_pairwise_governed"
    blended = real_rows + synth_rows + ext_rows
    rng.shuffle(blended)
    return blended[:total]


def load_external_pairwise_rows(
    report_path: Path,
    *,
    source_rows: list[dict[str, Any]],
    exclude_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = read_json(report_path)
    summary = {
        "report": rel(report_path),
        "report_present": report_path.exists(),
        "training_use_allowed": bool(report.get("training_use_allowed", False)),
        "loaded_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "rejection_reasons": Counter(),
    }
    if not report.get("training_use_allowed", False):
        summary["rejection_reasons"]["sampler_not_training_ready"] += 1
        return [], counter_to_plain(summary)
    artifact = ((report.get("artifacts") or {}).get("pairwise_training_jsonl") or "").strip()
    if not artifact:
        summary["rejection_reasons"]["missing_pairwise_artifact"] += 1
        return [], counter_to_plain(summary)
    path = ROOT / artifact
    if not path.exists():
        summary["rejection_reasons"]["pairwise_artifact_not_found"] += 1
        return [], counter_to_plain(summary)
    exclude_pairs = {pair_key(row) for row in source_rows + exclude_rows}
    exclude_sentences = {sentence_key(sentence) for row in source_rows + exclude_rows for sentence in row_sentences(row)}
    rows: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    for row in read_jsonl(path):
        summary["loaded_count"] += 1
        good = clean_sentence(row.get("sentence_good", ""))
        bad = clean_sentence(row.get("sentence_bad", ""))
        candidate = dict(row)
        candidate["sentence_good"] = good
        candidate["sentence_bad"] = bad
        key = pair_key(candidate)
        sent_keys = {sentence_key(good), sentence_key(bad)}
        if not good or not bad or good == bad:
            summary["rejection_reasons"]["invalid_pair"] += 1
            continue
        if key in exclude_pairs or key in seen_pairs:
            summary["rejection_reasons"]["pair_overlap_or_duplicate"] += 1
            continue
        if sent_keys & exclude_sentences:
            summary["rejection_reasons"]["sentence_overlap"] += 1
            continue
        candidate.setdefault("source", "external_open_sample_pairwise_distill")
        candidate.setdefault("generation_policy", "local_rule_corruption_no_external_inference")
        candidate["training_origin"] = "external_open_sample_pairwise_governed"
        rows.append(candidate)
        seen_pairs.add(key)
    summary["accepted_count"] = len(rows)
    summary["rejected_count"] = int(summary["loaded_count"]) - len(rows)
    return rows, counter_to_plain(summary)


def verify_dataset(
    *,
    synthetic_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    blend_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    exclude_rows: list[dict[str, Any]],
    targets: dict[str, list[str]],
    gates: dict[str, Any],
    max_synthetic_ratio: float,
    max_external_pair_ratio: float,
    external_summary: dict[str, Any],
) -> dict[str, Any]:
    exclude_pairs = {pair_key(row) for row in source_rows + exclude_rows}
    exclude_sentences = {sentence_key(sentence) for row in source_rows + exclude_rows for sentence in row_sentences(row)}
    synthetic_pairs = [pair_key(row) for row in synthetic_rows]
    synthetic_sentences = [sentence_key(sentence) for row in synthetic_rows for sentence in row_sentences(row)]
    pair_overlaps = sum(1 for key in synthetic_pairs if key in exclude_pairs)
    sentence_overlaps = sum(1 for key in synthetic_sentences if key in exclude_sentences)
    duplicate_pairs = len(synthetic_pairs) - len(set(synthetic_pairs))
    quality_scores = [float((row.get("quality") or {}).get("score") or 0.0) for row in synthetic_rows]
    mean_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    targeted = [row for row in synthetic_rows if is_targeted(row, targets)]
    synthetic_in_blend = sum(1 for row in blend_rows if row.get("training_origin") == "synthetic_verified_residual_targeted")
    external_in_blend = sum(1 for row in blend_rows if row.get("training_origin") == "external_open_sample_pairwise_governed")
    synthetic_ratio = synthetic_in_blend / len(blend_rows) if blend_rows else 0.0
    external_ratio = external_in_blend / len(blend_rows) if blend_rows else 0.0
    external_pairs = [pair_key(row) for row in external_rows]
    external_sentences = [sentence_key(sentence) for row in external_rows for sentence in row_sentences(row)]
    external_pair_overlaps = sum(1 for key in external_pairs if key in exclude_pairs)
    external_sentence_overlaps = sum(1 for key in external_sentences if key in exclude_sentences)
    by_rule = Counter(str(row.get("rule") or "unknown") for row in synthetic_rows)
    max_rule_share = max((count / max(1, len(synthetic_rows)) for count in by_rule.values()), default=0.0)
    checks = [
        check("accepted_synthetic_present", bool(synthetic_rows), f"accepted={len(synthetic_rows)}"),
        check("exact_pair_overlap_clear", pair_overlaps <= int(gates.get("max_exact_pair_overlap", 0)), f"overlaps={pair_overlaps}"),
        check(
            "exact_sentence_overlap_clear",
            sentence_overlaps <= int(gates.get("max_exact_sentence_overlap", 0)),
            f"overlaps={sentence_overlaps}",
        ),
        check("duplicate_pair_clear", duplicate_pairs == 0, f"duplicates={duplicate_pairs}"),
        check(
            "mean_quality_green",
            mean_quality >= float(gates.get("min_mean_quality_score", 0.76)),
            f"mean={mean_quality:.4f}",
        ),
        check(
            "targeted_share_green",
            (len(targeted) / max(1, len(synthetic_rows))) >= float(gates.get("min_targeted_share", 0.55)),
            f"targeted={len(targeted)}/{len(synthetic_rows)}",
        ),
        check(
            "synthetic_ratio_capped",
            synthetic_ratio <= max_synthetic_ratio + 1e-9,
            f"ratio={synthetic_ratio:.4f} max={max_synthetic_ratio:.4f}",
        ),
        check(
            "external_pair_ratio_capped",
            external_ratio <= max_external_pair_ratio + 1e-9,
            f"ratio={external_ratio:.4f} max={max_external_pair_ratio:.4f}",
        ),
        check(
            "external_pairs_leakage_clear",
            external_pair_overlaps == 0 and external_sentence_overlaps == 0,
            f"pair_overlaps={external_pair_overlaps} sentence_overlaps={external_sentence_overlaps}",
        ),
        check(
            "rule_diversity_cap",
            max_rule_share <= float(gates.get("max_rule_share", 0.18)) + 0.03,
            f"max_rule_share={max_rule_share:.4f}",
        ),
    ]
    return {
        "checks": checks,
        "passed": sum(1 for item in checks if item["passed"]),
        "total": len(checks),
        "ok": all(item["passed"] for item in checks),
        "pair_overlaps": pair_overlaps,
        "sentence_overlaps": sentence_overlaps,
        "duplicate_pairs": duplicate_pairs,
        "mean_quality_score": round(mean_quality, 4),
        "targeted_share": round(len(targeted) / max(1, len(synthetic_rows)), 4),
        "synthetic_ratio": round(synthetic_ratio, 4),
        "external_pair_ratio": round(external_ratio, 4),
        "external_pair_overlaps": external_pair_overlaps,
        "external_sentence_overlaps": external_sentence_overlaps,
        "external_summary": external_summary,
        "max_rule_share": round(max_rule_share, 4),
    }


def build_report(
    *,
    policy: dict[str, Any],
    residual: dict[str, Any],
    targets: dict[str, list[str]],
    source_rows: list[dict[str, Any]],
    exclude_rows: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    external_summary: dict[str, Any],
    blend_rows: list[dict[str, Any]],
    verification: dict[str, Any],
    out_data: Path,
    blend_out: Path,
    dataset_card_out: Path,
    dry_run: bool,
    seed: int,
) -> dict[str, Any]:
    by_rule = Counter(str(row.get("rule") or "unknown") for row in accepted)
    by_term = Counter(str(row.get("linguistics_term") or "unknown") for row in accepted)
    by_strategy = Counter(str(row.get("generation_strategy") or row.get("mutation_kind") or "unknown") for row in accepted)
    synthetic_in_blend = sum(1 for row in blend_rows if row.get("training_origin") == "synthetic_verified_residual_targeted")
    external_in_blend = sum(1 for row in blend_rows if row.get("training_origin") == "external_open_sample_pairwise_governed")
    return {
        "policy": "sparkstream_synthetic_data_curator_v0",
        "created_utc": now(),
        "mode": policy.get("default_mode", "residual_targeted_local"),
        "seed": seed,
        "training_ready": bool(verification.get("ok")),
        "dry_run": dry_run,
        "external_inference_calls": 0,
        "teacher_generation": "not_used",
        "methodology": {
            "source_research": [
                "Phi/Textbooks: small models can benefit from high-quality textbook/exercise data.",
                "Cosmopedia: prompt/topic diversity and duplicate control are first-order constraints.",
                "Self-Instruct/LLM2LLM: generation should be targeted and filtered, especially around errors.",
                "Model-collapse work: do not recursively train on untracked synthetic-only distributions.",
            ],
            "local_strategy": "residual_targeted_template_and_feature_preserving_mutation",
            "collapse_guard": policy.get("collapse_guard"),
        },
        "targets": targets,
        "source_rows": len(source_rows),
        "exclude_rows": len(exclude_rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "blend_count": len(blend_rows),
        "blend_synthetic_count": synthetic_in_blend,
        "blend_external_pair_count": external_in_blend,
        "blend_real_count": len(blend_rows) - synthetic_in_blend - external_in_blend,
        "blend_synthetic_ratio": round(synthetic_in_blend / max(1, len(blend_rows)), 4),
        "blend_external_pair_ratio": round(external_in_blend / max(1, len(blend_rows)), 4),
        "external_pair_count": len(external_rows),
        "external_pair_summary": external_summary,
        "by_rule": dict(by_rule.most_common()),
        "by_linguistics_term": dict(by_term.most_common()),
        "by_strategy": dict(by_strategy.most_common()),
        "verification": verification,
        "artifacts": {
            "synthetic_jsonl": rel(out_data),
            "blend_jsonl": rel(blend_out),
            "dataset_card": rel(dataset_card_out),
            "residual_escrow": "reports/residual_escrow.json",
        },
        "usage_policy": {
            "allowed_for_training": bool(verification.get("ok")),
            "allowed_profiles": ["inner_loop", "candidate", "seed_sweep"] if verification.get("ok") else [],
            "requires_public_private_delta_check": True,
            "requires_candidate_gate": True,
            "not_a_private_holdout": True,
            "external_pairs_low_ratio_only": True,
        },
        "residual_summary": residual.get("summary") or {},
    }


def build_dataset_card(report: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_card": "sparkstream_synthetic_babylm_residual_targeted_v0",
        "created_utc": report.get("created_utc"),
        "synthetic_jsonl": report["artifacts"]["synthetic_jsonl"],
        "blend_jsonl": report["artifacts"]["blend_jsonl"],
        "purpose": "Residual-targeted BabyLM training augmentation, not evaluation.",
        "source_data": (policy.get("babylm") or {}).get("sources", []),
        "excluded_data": (policy.get("babylm") or {}).get("excludes", []),
        "generation_method": report["methodology"]["local_strategy"],
        "external_inference_calls": 0,
        "teacher_generation": "not_used",
        "quality": report.get("verification"),
        "risks": [
            "Can overfit residual families if synthetic ratio is raised too high.",
            "External open-text pairwise rows are tiny, low-ratio training augmentation and not evaluation data.",
            "Can hide architecture weaknesses if used without ablation and public/private eval deltas.",
            "Should not be treated as a pristine holdout because it is generated for training intervention.",
        ],
        "governance": {
            "max_synthetic_ratio": (policy.get("collapse_guard") or {}).get("max_synthetic_ratio"),
            "max_external_pair_ratio": (policy.get("collapse_guard") or {}).get("max_external_pair_ratio"),
            "requires_holdout_exclusion": True,
            "requires_provenance": True,
            "requires_candidate_gate": True,
        },
    }


def is_targeted(row: dict[str, Any], targets: dict[str, list[str]]) -> bool:
    return str(row.get("rule") or "") in set(targets.get("rules") or []) or str(
        row.get("linguistics_term") or ""
    ) in set(targets.get("terms") or [])


def check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def interleave(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_len = max(len(a), len(b))
    for idx in range(max_len):
        if idx < len(a):
            rows.append(a[idx])
        if idx < len(b):
            rows.append(b[idx])
    return rows


def append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def clean_sentence(value: Any) -> str:
    return babylm_factory.clean_sentence(str(value))


def row_sentences(row: dict[str, Any]) -> list[str]:
    return [clean_sentence(row.get("sentence_good", "")), clean_sentence(row.get("sentence_bad", ""))]


def pair_key(row: dict[str, Any]) -> str:
    return hashlib.sha256("\n".join(row_sentences(row)).lower().encode("utf-8")).hexdigest()


def sentence_key(sentence: Any) -> str:
    return babylm_factory.sentence_key(clean_sentence(sentence))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def counter_to_plain(value: dict[str, Any]) -> dict[str, Any]:
    plain = dict(value)
    if isinstance(plain.get("rejection_reasons"), Counter):
        plain["rejection_reasons"] = dict(plain["rejection_reasons"])
    return plain


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
