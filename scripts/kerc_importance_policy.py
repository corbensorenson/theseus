#!/usr/bin/env python3
"""Source-visible calibrated importance policy for KERC residual allocation.

This module deliberately separates inference features from supervision. The
policy sees only source text and compiled packet structure. Labels are derived
from evaluator-only hard preservation constraints and are never exposed as
features. Fit, calibration, and final evaluation use the corpus's existing
source-group-disjoint train/dev/eval split.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Iterable

import numpy as np


POLICY = "project_theseus_kerc_calibrated_importance_policy_v1"
DIMENSIONS = ("semantic_importance", "surface_importance", "identity_anchoring")
FEATURE_NAMES = (
    "bias",
    "log1p_utf8_bytes",
    "log1p_word_count",
    "sentence_count",
    "digit_fraction",
    "uppercase_fraction",
    "quote_marker_count",
    "url_or_email_marker",
    "code_marker_count",
    "protected_object_count",
    "protected_quote_count",
    "protected_value_count",
    "protected_identity_count",
    "program_node_count",
    "program_negated_count",
    "program_nonasserted_modality_count",
    "correction_candidate_count",
)


class ImportancePolicyFault(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def source_visible_features(record: dict[str, Any]) -> np.ndarray:
    source = str(record.get("source_text") or "")
    packet = record.get("kernel_packet") if isinstance(record.get("kernel_packet"), dict) else {}
    program = packet.get("program") if isinstance(packet.get("program"), dict) else {}
    objects = packet.get("protected_objects") if isinstance(packet.get("protected_objects"), dict) else {}
    lattice = packet.get("correction_lattice") if isinstance(packet.get("correction_lattice"), dict) else {}
    nodes = program.get("nodes") if isinstance(program.get("nodes"), list) else []
    encoded = source.encode("utf-8")
    letters = [character for character in source if character.isalpha()]
    object_types = [str(row.get("object_type") or "") for row in objects.values() if isinstance(row, dict)]
    values = np.asarray(
        (
            1.0,
            math.log1p(len(encoded)),
            math.log1p(len(re.findall(r"\b\w+\b", source))),
            float(max(1, len(re.findall(r"[.!?]+(?:\s|$)", source)))),
            sum(character.isdigit() for character in source) / max(1, len(source)),
            sum(character.isupper() for character in letters) / max(1, len(letters)),
            float(sum(source.count(marker) for marker in ('"', "'", "“", "”", "‘", "’"))),
            float(bool(re.search(r"https?://|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", source))),
            float(bool(re.search(r"```|`[^`]+`|\b(?:def|class|fn|const|let|var)\b|[{};]", source))),
            float(len(objects)),
            float(sum(value == "QUOTE" for value in object_types)),
            float(sum(value in {"NUMBER", "DATE_TIME", "URL", "CODE", "FORMULA"} for value in object_types)),
            float(sum(value in {"PERSON", "PLACE", "ORGANIZATION", "IDENTIFIER"} for value in object_types)),
            float(len(nodes)),
            float(sum(str(node.get("polarity") or "") == "NEGATIVE" for node in nodes if isinstance(node, dict))),
            float(sum(str(node.get("modality") or "ASSERTED") != "ASSERTED" for node in nodes if isinstance(node, dict))),
            float(len(lattice.get("corrections") or [])),
        ),
        dtype=np.float64,
    )
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ImportancePolicyFault("invalid source-visible importance features")
    return values


def evaluator_only_targets(record: dict[str, Any]) -> np.ndarray:
    """Derive hard-preservation labels without leaking them into policy inputs."""

    packet = record["kernel_packet"]
    residual = packet["residual"]
    answer = record["answer_packet"]
    objects = packet.get("protected_objects") or {}
    object_types = {
        str(row.get("object_type") or "")
        for row in objects.values()
        if isinstance(row, dict)
    }
    claims = answer.get("claims") or []
    decision = answer.get("decision") or {}
    required_terms = answer.get("required_terms") or []
    required_caveats = answer.get("required_caveats") or []
    semantic = bool(
        len(claims) > 1
        or decision.get("disposition") in {"PARTIAL", "CLARIFY", "ABSTAIN"}
        or objects
        or required_terms
        or required_caveats
        or any(
            str(claim.get("polarity") or "") == "NEGATIVE"
            or str(claim.get("modality") or "ASSERTED") != "ASSERTED"
            or claim.get("attribution")
            for claim in claims
            if isinstance(claim, dict)
        )
    )
    surface = bool(
        objects
        or residual.get("token_tags")
        or required_terms
        or required_caveats
    )
    identity = bool(
        object_types
        & {"PERSON", "PLACE", "ORGANIZATION", "IDENTIFIER", "URL", "QUOTE"}
        or any(
            str(tag.get("tag") or "").startswith("ENTITY:")
            for tag in residual.get("token_tags") or []
            if isinstance(tag, dict)
        )
    )
    return np.asarray((semantic, surface, identity), dtype=np.float64)


def _fit_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int = 300,
    learning_rate: float = 0.08,
    l2: float = 1e-3,
) -> np.ndarray:
    weights = np.zeros(features.shape[1], dtype=np.float64)
    positive = float(targets.sum())
    negative = float(len(targets) - positive)
    if positive <= 0 or negative <= 0:
        raise ImportancePolicyFault("importance target lacks both classes")
    row_weights = np.where(
        targets > 0.5,
        len(targets) / (2.0 * positive),
        len(targets) / (2.0 * negative),
    )
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    for step in range(1, steps + 1):
        predictions = _sigmoid(features @ weights)
        gradient = features.T @ ((predictions - targets) * row_weights) / len(targets)
        gradient += l2 * np.r_[0.0, weights[1:]]
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return weights


def _temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    candidates = np.exp(np.linspace(math.log(0.25), math.log(4.0), 97))
    losses = []
    for value in candidates:
        probabilities = np.clip(_sigmoid(logits / value), 1e-7, 1.0 - 1e-7)
        losses.append(float(-np.mean(targets * np.log(probabilities) + (1.0 - targets) * np.log(1.0 - probabilities))))
    return float(candidates[int(np.argmin(losses))])


def _metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    predicted = probabilities >= 0.5
    expected = targets >= 0.5
    positive_recall = float(np.mean(predicted[expected])) if expected.any() else None
    negative_recall = float(np.mean(~predicted[~expected])) if (~expected).any() else None
    bins = np.minimum((probabilities * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(targets[mask].mean()))
    return {
        "row_count": int(len(targets)),
        "positive_prevalence": float(targets.mean()),
        "accuracy": float(np.mean(predicted == expected)),
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "weak_tail_recall": min(value for value in (positive_recall, negative_recall) if value is not None),
        "brier_score": float(np.mean(np.square(probabilities - targets))),
        "expected_calibration_error_10_bin": ece,
    }


def fit_importance_policy(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    by_split = {
        split: [row for row in rows if row.get("split") == split]
        for split in ("private_train", "private_dev", "private_eval")
    }
    if any(not values for values in by_split.values()):
        raise ImportancePolicyFault("importance fit requires train/dev/eval rows")
    group_sets = {
        split: {str(row["provenance"]["source_group"]) for row in values}
        for split, values in by_split.items()
    }
    if any(
        group_sets[left] & group_sets[right]
        for left, right in (("private_train", "private_dev"), ("private_train", "private_eval"), ("private_dev", "private_eval"))
    ):
        raise ImportancePolicyFault("importance split source groups overlap")
    matrices = {
        split: np.stack([source_visible_features(row) for row in values])
        for split, values in by_split.items()
    }
    targets = {
        split: np.stack([evaluator_only_targets(row) for row in values])
        for split, values in by_split.items()
    }
    train = matrices["private_train"]
    mean = train[:, 1:].mean(axis=0)
    scale = train[:, 1:].std(axis=0)
    scale[scale < 1e-8] = 1.0

    def normalize(matrix: np.ndarray) -> np.ndarray:
        output = matrix.copy()
        output[:, 1:] = (output[:, 1:] - mean) / scale
        return output

    normalized = {split: normalize(matrix) for split, matrix in matrices.items()}
    weights = []
    temperatures = []
    metrics: dict[str, dict[str, Any]] = {split: {} for split in by_split}
    for index, name in enumerate(DIMENSIONS):
        fitted = _fit_logistic(normalized["private_train"], targets["private_train"][:, index])
        dev_logits = normalized["private_dev"] @ fitted
        temperature = _temperature(dev_logits, targets["private_dev"][:, index])
        weights.append(fitted.tolist())
        temperatures.append(temperature)
        for split in by_split:
            probabilities = _sigmoid((normalized[split] @ fitted) / temperature)
            metrics[split][name] = _metrics(probabilities, targets[split][:, index])
    policy = {
        "policy": POLICY,
        "feature_names": list(FEATURE_NAMES),
        "inference_feature_authority": "source_text_and_compiled_packet_only",
        "evaluator_only_target_authority": "hard_preservation_constraints_not_visible_to_policy",
        "fit_split": "private_train",
        "calibration_split": "private_dev",
        "final_evaluation_split": "private_eval",
        "source_group_disjoint": True,
        "normalization_mean_without_bias": mean.tolist(),
        "normalization_scale_without_bias": scale.tolist(),
        "weights_by_dimension": dict(zip(DIMENSIONS, weights)),
        "temperature_by_dimension": dict(zip(DIMENSIONS, temperatures)),
        "metrics_by_split": metrics,
        "capability_or_rate_distortion_utility_claim": False,
        "fallback_return_count": 0,
    }
    policy["policy_sha256"] = _digest(policy)
    return policy


def predict_importance(record: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    core = {key: copy.deepcopy(value) for key, value in policy.items() if key != "policy_sha256"}
    if policy.get("policy") != POLICY or policy.get("policy_sha256") != _digest(core):
        raise ImportancePolicyFault("importance policy identity mismatch")
    if policy.get("feature_names") != list(FEATURE_NAMES):
        raise ImportancePolicyFault("importance feature contract mismatch")
    features = source_visible_features(record)
    mean = np.asarray(policy["normalization_mean_without_bias"], dtype=np.float64)
    scale = np.asarray(policy["normalization_scale_without_bias"], dtype=np.float64)
    features[1:] = (features[1:] - mean) / scale
    probabilities = {}
    for name in DIMENSIONS:
        weights = np.asarray(policy["weights_by_dimension"][name], dtype=np.float64)
        temperature = float(policy["temperature_by_dimension"][name])
        probabilities[name] = float(_sigmoid(np.asarray([(features @ weights) / temperature]))[0])
    receipt = {
        "policy": POLICY,
        "policy_sha256": policy["policy_sha256"],
        "source_visible_features_sha256": _digest(source_visible_features(record).tolist()),
        "scores": probabilities,
        "allocation_importance": max(probabilities.values()),
        "target_fields_visible_to_policy": [],
        "fallback_return_count": 0,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt
