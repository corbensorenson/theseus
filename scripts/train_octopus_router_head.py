"""Train a local sparse Octopus Router head from routing traces.

This is intentionally small, deterministic, and dependency-free. The goal is
to turn ORA routing traces into a learned, inspectable router artifact before
heavy model training begins. It does not call external inference providers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REAL_TRACE_PATH = "reports/routing_memory_real_traces.jsonl"

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "for",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-eval", default="reports/octopus_router_eval.json")
    parser.add_argument("--arm-registry", default="reports/arm_registry.json")
    parser.add_argument("--dataset-out", default="reports/octopus_router_trace_dataset.json")
    parser.add_argument("--model-out", default="reports/octopus_router_head_model.json")
    parser.add_argument("--eval-out", default="reports/octopus_router_head_eval.json")
    parser.add_argument("--out", default="reports/octopus_router_head_report.json")
    parser.add_argument(
        "--extra-traces",
        action="append",
        default=None,
        help="Optional JSONL workflow routing traces from real local ratchet runs.",
    )
    parser.add_argument("--min-contrastive-accuracy", type=float, default=0.95)
    args = parser.parse_args()
    if args.extra_traces is None:
        args.extra_traces = [DEFAULT_REAL_TRACE_PATH] if Path(DEFAULT_REAL_TRACE_PATH).exists() else []

    router_eval = read_json(args.router_eval, {})
    arm_registry = read_json(args.arm_registry, {})
    arms = arm_registry.get("arms", [])
    examples = build_trace_dataset(router_eval, arms)
    extra_examples = build_extra_trace_dataset(args.extra_traces, arms)
    examples.extend(extra_examples)
    contrastive_negatives = build_contrastive_negatives(examples)
    train_examples = [row for row in examples if row["split"] == "train"]
    holdout_examples = [row for row in examples if row["split"] == "holdout"]

    model = train_centroid_router(train_examples, arms)
    model["contrastive_negative_policy"] = {
        "method": "hard_wrong_labelset_margin_audit_v1",
        "negative_count": len(contrastive_negatives),
        "min_contrastive_accuracy": args.min_contrastive_accuracy,
    }
    evaluation = evaluate_model(model, holdout_examples, contrastive_negatives)
    report = build_report(
        args=args,
        model=model,
        evaluation=evaluation,
        examples=examples,
        contrastive_negatives=contrastive_negatives,
        router_eval=router_eval,
    )

    write_json(args.dataset_out, dataset_payload(examples, contrastive_negatives))
    write_json(args.model_out, model)
    write_json(args.eval_out, evaluation)
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def build_trace_dataset(router_eval: dict[str, Any], arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arm_keywords = {
        arm["arm_name"]: [str(keyword).lower() for keyword in arm.get("routing_keywords", [])]
        for arm in arms
    }
    examples: list[dict[str, Any]] = []
    for row in router_eval.get("decisions", []):
        labels = sorted(set(row.get("expected", [])))
        if row.get("risk") in ("high", "critical"):
            labels = sorted(set(labels + ["safety_reflex_arm"]))
        variants = augment_task(row)
        for idx, text in enumerate(variants):
            split = "holdout" if idx == len(variants) - 1 else "train"
            examples.append(
                {
                    "trace_id": stable_id(row.get("task_id", "route"), idx, text),
                    "source_task_id": row.get("task_id"),
                    "split": split,
                    "task": text,
                    "risk": row.get("risk", "low"),
                    "routing_pattern": row.get("pattern") or row.get("routing_pattern"),
                    "expected_arms": labels,
                    "features": sorted(
                        featurize(
                            text=text,
                            risk=row.get("risk", "low"),
                            pattern=row.get("pattern") or row.get("routing_pattern", "single"),
                            arm_keywords=arm_keywords,
                        )
                    ),
                }
            )
    return examples


def build_extra_trace_dataset(paths: list[str], arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arm_keywords = {
        arm["arm_name"]: [str(keyword).lower() for keyword in arm.get("routing_keywords", [])]
        for arm in arms
    }
    examples: list[dict[str, Any]] = []
    for path in paths:
        file = Path(path)
        if not file.exists():
            continue
        for line_no, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            labels = sorted(set(row.get("expected_arms") or row.get("selected_arms") or []))
            if not labels:
                continue
            risk = row.get("risk", "low")
            pattern = row.get("routing_pattern") or row.get("pattern", "single")
            task = str(row.get("task") or row.get("workflow") or row.get("command") or "").strip()
            if not task:
                continue
            if risk in ("high", "critical"):
                labels = sorted(set(labels + ["safety_reflex_arm"]))
            schema_bound = schema_bound_real_trace(row, labels)
            examples.append(
                {
                    "trace_id": stable_id(path, line_no, task),
                    "source_task_id": row.get("trace_id") or row.get("task_id") or f"{file.stem}:{line_no}",
                    "split": row.get("split", "train"),
                    "task": task,
                    "risk": risk,
                    "routing_pattern": pattern,
                    "expected_arms": labels,
                    "schema_bound": schema_bound,
                    "outcome_ok": bool((row.get("outcome") or {}).get("ok", True)) if isinstance(row.get("outcome"), dict) else None,
                    "features": sorted(
                        featurize(
                            text=task,
                            risk=risk,
                            pattern=pattern,
                            arm_keywords=arm_keywords,
                        )
                    ),
                    "source": "real_workflow_trace",
                }
            )
    return examples


def schema_bound_real_trace(row: dict[str, Any], labels: list[str]) -> bool:
    envelopes = row.get("permission_envelopes")
    if not isinstance(envelopes, dict) or not envelopes:
        return False
    selected = set(labels)
    if not selected:
        return False
    if not selected.issubset(set(str(key) for key in envelopes)):
        return False
    for arm_name in selected:
        envelope = envelopes.get(arm_name)
        if not isinstance(envelope, dict):
            return False
        if envelope.get("external_inference") != "forbidden":
            return False
        if "runtime_tier" not in envelope or "tools" not in envelope:
            return False
    return bool(row.get("trace_id") and row.get("task") and row.get("source"))


def augment_task(row: dict[str, Any]) -> list[str]:
    task = str(row.get("task", "")).strip()
    compact = " ".join(token for token in tokenize(task) if token not in STOPWORDS)
    domain_terms = " ".join(key_terms(task)[:8])
    risk = row.get("risk", "low")
    pattern = row.get("pattern") or row.get("routing_pattern", "single")
    return [
        task,
        task.lower(),
        f"Route this {risk} task through the right specialist arms: {task}",
        f"Specialist dispatch request; pattern={pattern}; signals={domain_terms}; task={task}",
        f"ORA dispatch for {risk} {pattern} work: {compact}",
    ]


def key_terms(text: str) -> list[str]:
    terms = []
    for token in tokenize(text):
        if token not in STOPWORDS and len(token) > 2 and token not in terms:
            terms.append(token)
    return terms


def featurize(
    *,
    text: str,
    risk: str,
    pattern: str,
    arm_keywords: dict[str, list[str]],
) -> Counter[str]:
    tokens = [token for token in tokenize(text) if token not in STOPWORDS]
    features: Counter[str] = Counter()
    features["bias"] = 1
    features[f"risk:{risk}"] += 3
    features[f"pattern:{pattern}"] += 2
    if risk in ("high", "critical"):
        features["risk:safety_required"] += 4
    for token in tokens:
        features[f"tok:{token}"] += 1
    for left, right in zip(tokens, tokens[1:]):
        features[f"bi:{left}_{right}"] += 1
    text_l = text.lower()
    for arm_name, keywords in arm_keywords.items():
        for keyword in keywords:
            if keyword and keyword in text_l:
                features[f"arm_keyword:{arm_name}"] += 2
    return features


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def train_centroid_router(examples: list[dict[str, Any]], arms: list[dict[str, Any]]) -> dict[str, Any]:
    centroids: dict[str, Counter[str]] = defaultdict(Counter)
    counts: Counter[str] = Counter()
    for row in examples:
        label_key = label_key_for(row["expected_arms"])
        vector = Counter({feature: 1.0 for feature in row.get("features", [])})
        centroids[label_key].update(vector)
        counts[label_key] += 1

    normalized = {}
    for label_key, vector in centroids.items():
        averaged = Counter({feature: value / counts[label_key] for feature, value in vector.items()})
        normalized[label_key] = dict(l2_normalize(averaged))

    arm_names = sorted(arm["arm_name"] for arm in arms if arm["arm_name"] != "head_router")
    arm_keywords = {
        arm["arm_name"]: [str(keyword).lower() for keyword in arm.get("routing_keywords", [])]
        for arm in arms
    }
    return {
        "policy": "local_only_no_external_inference",
        "framework": "octopus_router_head",
        "model_type": "sparse_centroid_multilabel_router_v0",
        "training_method": "deterministic_centroids_over_augmented_local_routing_traces_with_contrastive_negative_audit",
        "arm_names": arm_names,
        "arm_keywords": arm_keywords,
        "labelsets": sorted(normalized),
        "centroids": normalized,
        "feature_schema": [
            "bias",
            "risk:<tier>",
            "pattern:<routing_pattern>",
            "risk:safety_required",
            "tok:<token>",
            "bi:<token>_<token>",
            "arm_keyword:<arm_name>",
        ],
        "external_inference_calls": 0,
    }


def build_contrastive_negatives(examples: list[dict[str, Any]], *, max_per_example: int = 3) -> list[dict[str, Any]]:
    labelsets = sorted({label_key_for(row["expected_arms"]) for row in examples if row.get("expected_arms")})
    negatives: list[dict[str, Any]] = []
    for row in examples:
        positive_key = label_key_for(row["expected_arms"])
        positive = set(label_key_to_arms(positive_key))
        if not positive:
            continue
        candidates: list[tuple[tuple[int, int, int, str], str]] = []
        positive_safety = "safety_reflex_arm" in positive
        for label_key in labelsets:
            if label_key == positive_key:
                continue
            negative = set(label_key_to_arms(label_key))
            same_safety = int(("safety_reflex_arm" in negative) == positive_safety)
            shared = len(positive & negative)
            size_closeness = -abs(len(positive) - len(negative))
            candidates.append(((same_safety, shared, size_closeness, label_key), label_key))
        candidates.sort(reverse=True)
        for rank, (_score, label_key) in enumerate(candidates[:max_per_example], start=1):
            negatives.append(
                {
                    "contrastive_id": stable_id("octopus_router_contrastive", row["trace_id"], label_key),
                    "positive_trace_id": row["trace_id"],
                    "source_task_id": row.get("source_task_id"),
                    "split": row.get("split"),
                    "task": row.get("task"),
                    "risk": row.get("risk"),
                    "routing_pattern": row.get("routing_pattern"),
                    "features": row.get("features", []),
                    "positive_arms": sorted(positive),
                    "negative_arms": label_key_to_arms(label_key),
                    "negative_label_key": label_key,
                    "hardness_rank": rank,
                    "shared_arm_count": len(positive & set(label_key_to_arms(label_key))),
                    "source": "contrastive_negative_wrong_labelset",
                    "public_training_rows_written": 0,
                    "external_inference_calls": 0,
                    "fallback_return_count": 0,
                }
            )
    return negatives


def evaluate_model(
    model: dict[str, Any],
    examples: list[dict[str, Any]],
    contrastive_negatives: list[dict[str, Any]],
) -> dict[str, Any]:
    decisions = []
    true_positive: Counter[str] = Counter()
    false_positive: Counter[str] = Counter()
    false_negative: Counter[str] = Counter()
    exact = 0
    risk_passed = 0
    for row in examples:
        predicted = predict(model, row["task"], row["risk"], row["routing_pattern"])
        expected = sorted(set(row["expected_arms"]))
        predicted_set = set(predicted)
        expected_set = set(expected)
        for arm in predicted_set & expected_set:
            true_positive[arm] += 1
        for arm in predicted_set - expected_set:
            false_positive[arm] += 1
        for arm in expected_set - predicted_set:
            false_negative[arm] += 1
        exact_match = predicted_set == expected_set
        if exact_match:
            exact += 1
        risk_ok = row["risk"] not in ("high", "critical") or "safety_reflex_arm" in predicted_set
        if risk_ok:
            risk_passed += 1
        decisions.append(
            {
                "trace_id": row["trace_id"],
                "source_task_id": row["source_task_id"],
                "risk": row["risk"],
                "routing_pattern": row["routing_pattern"],
                "expected_arms": expected,
                "predicted_arms": predicted,
                "exact_match": exact_match,
                "risk_routing_passed": risk_ok,
            }
        )

    tp = sum(true_positive.values())
    fp = sum(false_positive.values())
    fn = sum(false_negative.values())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    per_arm = {}
    for arm in sorted(set(true_positive) | set(false_positive) | set(false_negative)):
        arm_tp = true_positive[arm]
        arm_fp = false_positive[arm]
        arm_fn = false_negative[arm]
        arm_precision = arm_tp / max(1, arm_tp + arm_fp)
        arm_recall = arm_tp / max(1, arm_tp + arm_fn)
        per_arm[arm] = {
            "precision": round(arm_precision, 4),
            "recall": round(arm_recall, 4),
            "tp": arm_tp,
            "fp": arm_fp,
            "fn": arm_fn,
        }
    exact_accuracy = exact / max(1, len(examples))
    risk_accuracy = risk_passed / max(1, len(examples))
    contrastive = evaluate_contrastive_negatives(model, contrastive_negatives, split="holdout")
    promotion_gate_passed = (
        exact_accuracy >= 0.95
        and risk_accuracy == 1.0
        and contrastive["contrastive_holdout_negatives"] > 0
        and contrastive["contrastive_negative_accuracy"] >= float(model.get("contrastive_negative_policy", {}).get("min_contrastive_accuracy", 0.95))
    )
    return {
        "policy": "local_only_no_external_inference",
        "methodology": "octopus_router_head_holdout_eval",
        "metrics": {
            "holdout_cases": len(examples),
            "exact_set_accuracy": round(exact_accuracy, 4),
            "arm_micro_precision": round(precision, 4),
            "arm_micro_recall": round(recall, 4),
            "arm_micro_f1": round(f1, 4),
            "risk_routing_accuracy": round(risk_accuracy, 4),
            **contrastive,
            "external_inference_calls": 0,
        },
        "per_arm": per_arm,
        "decisions": decisions,
        "contrastive_decisions": contrastive_decisions(model, contrastive_negatives, split="holdout")[:80],
        "promotion_gate_passed": promotion_gate_passed,
        "external_inference_calls": 0,
    }


def evaluate_contrastive_negatives(model: dict[str, Any], negatives: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    rows = [row for row in negatives if row.get("split") == split]
    margins: list[float] = []
    passed = 0
    for row in rows:
        positive_key = label_key_for(row.get("positive_arms", []))
        negative_key = str(row.get("negative_label_key") or label_key_for(row.get("negative_arms", [])))
        scores = labelset_scores_for_features(model, row.get("features", []))
        margin = scores.get(positive_key, -1.0) - scores.get(negative_key, -1.0)
        margins.append(margin)
        if margin > 0.0:
            passed += 1
    accuracy = passed / max(1, len(rows))
    return {
        "contrastive_holdout_negatives": len(rows),
        "contrastive_negative_passed": passed,
        "contrastive_negative_accuracy": round(accuracy, 4),
        "contrastive_min_margin": round(min(margins), 6) if margins else 0.0,
        "contrastive_mean_margin": round(sum(margins) / len(margins), 6) if margins else 0.0,
    }


def contrastive_decisions(model: dict[str, Any], negatives: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in [item for item in negatives if item.get("split") == split]:
        positive_key = label_key_for(row.get("positive_arms", []))
        negative_key = str(row.get("negative_label_key") or label_key_for(row.get("negative_arms", [])))
        scores = labelset_scores_for_features(model, row.get("features", []))
        positive_score = scores.get(positive_key, -1.0)
        negative_score = scores.get(negative_key, -1.0)
        decisions.append(
            {
                "contrastive_id": row.get("contrastive_id"),
                "positive_trace_id": row.get("positive_trace_id"),
                "split": row.get("split"),
                "positive_arms": row.get("positive_arms"),
                "negative_arms": row.get("negative_arms"),
                "positive_score": round(positive_score, 6),
                "negative_score": round(negative_score, 6),
                "margin": round(positive_score - negative_score, 6),
                "passed": positive_score > negative_score,
            }
        )
    return decisions


def predict(model: dict[str, Any], task: str, risk: str, pattern: str) -> list[str]:
    vector = l2_normalize(featurize(text=task, risk=risk, pattern=pattern, arm_keywords=model.get("arm_keywords", {})))
    scores = labelset_scores(model, vector)
    best_label = ""
    best_score = -1.0
    for label_key, score in scores.items():
        if score > best_score:
            best_label = label_key
            best_score = score
    arms = sorted(label_key_to_arms(best_label))
    if risk in ("high", "critical") and "safety_reflex_arm" not in arms:
        arms.append("safety_reflex_arm")
    return sorted(set(arms))


def labelset_scores_for_features(model: dict[str, Any], features: list[str]) -> dict[str, float]:
    vector = l2_normalize(Counter({feature: 1.0 for feature in features}))
    return labelset_scores(model, vector)


def labelset_scores(model: dict[str, Any], vector: dict[str, float]) -> dict[str, float]:
    return {label_key: dot(vector, centroid) for label_key, centroid in model.get("centroids", {}).items()}


def label_key_for(arms: list[str]) -> str:
    return "|".join(sorted(set(arms)))


def label_key_to_arms(label_key: str) -> list[str]:
    return [part for part in label_key.split("|") if part]


def l2_normalize(vector: Counter[str] | dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector.values()))
    if norm <= 0.0:
        return {}
    return {feature: float(value) / norm for feature, value in vector.items()}


def dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * float(right.get(feature, 0.0)) for feature, value in left.items())


def dataset_payload(examples: list[dict[str, Any]], contrastive_negatives: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "policy": "local_only_no_external_inference",
        "framework": "octopus_router_trace_dataset",
        "summary": {
            "examples": len(examples),
            "train": sum(1 for row in examples if row["split"] == "train"),
            "holdout": sum(1 for row in examples if row["split"] == "holdout"),
            "real_trace_examples": sum(1 for row in examples if row.get("source") == "real_workflow_trace"),
            "schema_bound_real_trace_examples": sum(1 for row in examples if row.get("source") == "real_workflow_trace" and row.get("schema_bound") is True),
            "contrastive_negatives": len(contrastive_negatives),
            "contrastive_holdout_negatives": sum(1 for row in contrastive_negatives if row.get("split") == "holdout"),
            "labelsets": sorted({label_key_for(row["expected_arms"]) for row in examples}),
        },
        "examples": examples,
        "contrastive_negatives": contrastive_negatives,
        "external_inference_calls": 0,
    }


def build_report(
    *,
    args: argparse.Namespace,
    model: dict[str, Any],
    evaluation: dict[str, Any],
    examples: list[dict[str, Any]],
    contrastive_negatives: list[dict[str, Any]],
    router_eval: dict[str, Any],
) -> dict[str, Any]:
    metrics = evaluation.get("metrics", {})
    trace_summary = {
        "source_cases": len(router_eval.get("decisions", [])),
        "augmented_examples": len(examples),
        "train_examples": sum(1 for row in examples if row["split"] == "train"),
        "holdout_examples": sum(1 for row in examples if row["split"] == "holdout"),
        "real_trace_examples": sum(1 for row in examples if row.get("source") == "real_workflow_trace"),
        "schema_bound_real_trace_examples": sum(1 for row in examples if row.get("source") == "real_workflow_trace" and row.get("schema_bound") is True),
        "contrastive_negatives": len(contrastive_negatives),
        "contrastive_holdout_negatives": sum(1 for row in contrastive_negatives if row.get("split") == "holdout"),
        "labelsets": len(model.get("labelsets", [])),
    }
    promotion_gate_passed = bool(evaluation.get("promotion_gate_passed")) and trace_summary["schema_bound_real_trace_examples"] > 0
    records = build_viea_router_head_records(
        trace_summary=trace_summary,
        metrics=metrics,
        artifacts={
            "dataset": args.dataset_out,
            "model": args.model_out,
            "eval": args.eval_out,
            "router_eval": args.router_eval,
            "arm_registry": args.arm_registry,
            "extra_traces": args.extra_traces,
        },
        promotion_gate_passed=promotion_gate_passed,
    )
    return {
        "policy": "local_only_no_external_inference",
        "framework": "octopus_router_head_training",
        "status": "trained_contrastive_ready" if promotion_gate_passed else "needs_more_traces_or_contrastive_margin",
        "model_type": model["model_type"],
        "training_source": args.router_eval,
        "artifacts": {
            "dataset": args.dataset_out,
            "model": args.model_out,
            "eval": args.eval_out,
            "router_eval": args.router_eval,
            "arm_registry": args.arm_registry,
            "extra_traces": args.extra_traces,
        },
        "trace_summary": trace_summary,
        "metrics": metrics,
        "promotion_gate_passed": promotion_gate_passed,
        "learned_generation_claim_allowed": False,
        "candidate_generation_credit": 0,
        "router_selection_only": True,
        "viea_router_head_records": records,
        "next_actions": next_actions(evaluation),
        "external_inference_calls": 0,
    }


def build_viea_router_head_records(
    *,
    trace_summary: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    promotion_gate_passed: bool,
) -> list[dict[str, Any]]:
    route_id = "octopus_router_head.training_v1"
    support_state = "SUPPORTED" if promotion_gate_passed else "RESIDUAL_REVIEW"
    common = {
        "route_id": route_id,
        "task_kind": "moecot_learned_router_head_training",
        "target": "octopus_router_head",
        "support_state": support_state,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
    }
    blocked_reason = "none" if promotion_gate_passed else "missing_real_schema_traces_or_contrastive_margin"
    return [
        {
            **common,
            "record_type": "routing_decision",
            "record_id": stable_id("router_head_route_decision", metrics, trace_summary),
            "arm_id": "head_router",
            "selected_route": "rule_bootloader_plus_sparse_head_candidate",
            "route_state": "candidate_head_passed_gate" if promotion_gate_passed else "candidate_head_blocked",
            "task_fit": "specialist_arm_selection",
            "schema_bound_real_trace_examples": trace_summary.get("schema_bound_real_trace_examples"),
            "contrastive_negative_count": trace_summary.get("contrastive_negatives"),
        },
        {
            **common,
            "record_type": "authority_use_receipt",
            "record_id": stable_id("router_head_authority", artifacts),
            "authority_scope": "read_router_eval_read_real_route_traces_write_router_head_artifacts",
            "allowed_effects": ["read_local_reports", "read_metadata_route_traces", "write_router_head_reports"],
            "denied_effects": ["runtime_external_inference", "public_benchmark_training", "learned_generation_credit"],
        },
        {
            **common,
            "record_type": "resource_budget",
            "record_id": stable_id("router_head_resource", trace_summary),
            "worker_limit": 1,
            "estimated_latency_ms": 0,
            "candidate_count": trace_summary.get("augmented_examples"),
        },
        {
            **common,
            "record_type": "generation_mode",
            "record_id": stable_id("router_head_generation_mode", route_id),
            "mode": "router_head_selection_not_generation",
            "candidate_generation_credit": 0,
            "learned_generation_claim_allowed": False,
            "fallback_return_used": False,
        },
        {
            **common,
            "record_type": "failure_boundary",
            "record_id": stable_id("router_head_failure_boundary", blocked_reason),
            "failure_id": stable_id("router_head_failure", blocked_reason),
            "blocked_reason": blocked_reason,
            "terminal": promotion_gate_passed,
            "structured_non_solved": not promotion_gate_passed,
        },
        {
            **common,
            "record_type": "artifact_graph_record",
            "record_id": stable_id("router_head_artifact_graph", artifacts),
            "artifact_ref": artifacts.get("model"),
            "evidence_ref": artifacts.get("eval"),
            "source_refs": [artifacts.get("router_eval"), artifacts.get("arm_registry"), *artifacts.get("extra_traces", [])],
            "claim_refs": ["octopus_router_head_contrastive_training_gate"],
            "replay_grade": "metadata_replayable",
            "provenance_status": "local_metadata_only",
            "non_claims": ["not learned code generation", "not public benchmark evidence"],
        },
        {
            **common,
            "record_type": "claim_record",
            "record_id": stable_id("router_head_claim", metrics, promotion_gate_passed),
            "claim_id": "octopus_router_head_contrastive_training_gate",
            "support_state": support_state,
            "evidence_ref": artifacts.get("eval"),
            "learned_generation_claim_allowed": False,
        },
        {
            **common,
            "record_type": "evidence_transition_record",
            "record_id": stable_id("router_head_evidence_transition", support_state),
            "previous_support_state": "RULE_BOOTLOADER_ONLY",
            "current_support_state": support_state,
            "evidence_ref": artifacts.get("eval"),
        },
        {
            **common,
            "record_type": "residual_record",
            "record_id": stable_id("router_head_residual", blocked_reason),
            "support_state": "NONE" if promotion_gate_passed else "RESIDUAL_REVIEW",
            "blocked_reason": blocked_reason,
            "residuals": [] if promotion_gate_passed else ["collect_more_schema_bound_real_route_traces", "improve_contrastive_margin"],
        },
    ]


def next_actions(evaluation: dict[str, Any]) -> list[str]:
    if evaluation.get("promotion_gate_passed"):
        return [
            "Use the learned sparse head as the router training artifact while keeping the rule router as a deterministic bootloader/fallback.",
            "Append future real task-to-arm traces and retrain this head before every major architecture gate.",
            "Keep contrastive negative margins in the router-head gate as the arm ecosystem grows.",
        ]
    return [
        "Add more local routing traces before promoting the learned router head.",
        "Inspect holdout routing misses and add bridge routing cases for confused arm boundaries.",
    ]


def stable_id(*parts: Any) -> str:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:16]


def read_json(path: str, default: Any) -> Any:
    file = Path(path)
    if not file.exists():
        return default
    return json.loads(file.read_text(encoding="utf-8"))


def write_json(path: str, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
