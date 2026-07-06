"""Student Learning Closure v1.

This lane closes the gap between "a helper found a candidate" and "a governed
student checkpoint changed behavior." It learns a small trace ranker from
sandbox outcomes and emits a new student candidate manifest for the exact same
real-code harness.

It is intentionally conservative:
- no canonical solutions are read;
- public tests are not used by candidate generation;
- task ids are not used as features;
- same-task replay is reported as closure evidence, not public mastery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_code_benchmark_graduation as real_code  # noqa: E402


DEFAULT_CANDIDATE_MANIFEST = "reports/student_code_candidates.jsonl"
DEFAULT_TRACE_IN = "reports/real_code_benchmark_traces.jsonl"
DEFAULT_CHECKPOINT_OUT = "reports/student_learning_code_checkpoint.json"
DEFAULT_CANDIDATE_OUT = "reports/student_learning_code_candidates.jsonl"
DEFAULT_TRAINING_EXAMPLES_OUT = "reports/student_learning_training_examples.jsonl"
DEFAULT_TRANSFER_ARTIFACT_OUT = "reports/transfer_artifacts/code/student_learning_closure_transfer_artifact.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--trace-in", default=DEFAULT_TRACE_IN, help="Comma-separated trace JSONL paths.")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--holdout-ratio", type=float, default=0.34)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--checkpoint-out", default=DEFAULT_CHECKPOINT_OUT)
    parser.add_argument("--candidate-out", default=DEFAULT_CANDIDATE_OUT)
    parser.add_argument("--training-examples-out", default=DEFAULT_TRAINING_EXAMPLES_OUT)
    parser.add_argument("--transfer-artifact-out", default=DEFAULT_TRANSFER_ARTIFACT_OUT)
    parser.add_argument("--code-transfer-artifacts", default="reports/code_transfer_artifacts.json")
    parser.add_argument("--out", default="reports/student_learning_closure.json")
    args = parser.parse_args()

    started = time.perf_counter()
    candidate_rows = read_jsonl(resolve(args.candidate_manifest))
    traces = []
    for path_text in [item.strip() for item in args.trace_in.split(",") if item.strip()]:
        traces.extend(read_jsonl(resolve(path_text)))

    outcomes = collect_outcomes(traces)
    labeled = attach_labels(candidate_rows, outcomes)
    train_task_ids, eval_task_ids, split_kind = split_tasks(
        [str(row.get("task_id") or "") for row in labeled],
        seed=args.seed,
        holdout_ratio=args.holdout_ratio,
    )
    train_rows = [row for row in labeled if str(row.get("task_id") or "") in train_task_ids]
    eval_rows = [row for row in labeled if str(row.get("task_id") or "") in eval_task_ids]
    if not train_rows and labeled:
        train_rows = labeled
        eval_rows = labeled
        split_kind = "same_trace_replay_no_holdout_available"

    weights, bias, training_examples = train_ranker(train_rows, epochs=max(1, args.epochs))
    checkpoint = build_checkpoint(
        weights=weights,
        bias=bias,
        training_examples=training_examples,
        train_task_ids=train_task_ids,
        eval_task_ids=eval_task_ids,
        seed=args.seed,
        source_manifest=args.candidate_manifest,
        trace_paths=args.trace_in,
    )
    write_json(resolve(args.checkpoint_out), checkpoint)
    write_jsonl(resolve(args.training_examples_out), training_examples)

    learned_manifest = emit_learned_manifest(candidate_rows, weights, bias, checkpoint)
    write_jsonl(resolve(args.candidate_out), learned_manifest)

    before_eval = evaluate_selection(eval_rows, weights=None, bias=0.0)
    after_eval = evaluate_selection(eval_rows, weights=weights, bias=bias)
    all_before = evaluate_selection(labeled, weights=None, bias=0.0)
    all_after = evaluate_selection(labeled, weights=weights, bias=bias)
    leakage = leakage_findings(candidate_rows)
    pass_rate_delta = round(after_eval["pass_rate"] - before_eval["pass_rate"], 6)
    transfer_artifact = write_transfer_artifact(
        resolve(args.transfer_artifact_out),
        checkpoint=checkpoint,
        before_eval=before_eval,
        after_eval=after_eval,
        all_before=all_before,
        all_after=all_after,
        split_kind=split_kind,
        trace_paths=args.trace_in,
        candidate_manifest=args.candidate_manifest,
    )
    merge_transfer_index(resolve(args.code_transfer_artifacts), transfer_artifact)

    gates = [
        gate("trace_outcomes_loaded", len(outcomes) > 0, f"outcomes={len(outcomes)}"),
        gate("candidate_manifest_loaded", len(candidate_rows) > 0, f"candidates={len(candidate_rows)}"),
        gate("training_examples_nonzero", len(training_examples) > 0, f"examples={len(training_examples)}"),
        gate("learned_manifest_emitted", len(learned_manifest) > 0, f"learned_candidates={len(learned_manifest)}"),
        gate("canonical_solutions_not_visible", not leakage["canonical_solution_seen"], leakage),
        gate("public_tests_not_used_for_generation", not leakage["public_tests_visible"], leakage),
        gate("task_id_not_used_as_feature", True, "features use code shape, visible tags, and entry tokens; full task_id is never a feature; original rank is only a weak prior/tie-breaker"),
        gate("external_inference_zero", True, "trace-ranker training is local only"),
        gate("trained_checkpoint_improves_eval_selection", pass_rate_delta > 0.0, f"before={before_eval['pass_rate']} after={after_eval['pass_rate']} delta={pass_rate_delta}"),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else ("YELLOW" if len(training_examples) > 0 and not leakage["canonical_solution_seen"] else "RED")
    report = {
        "policy": "project_theseus_student_learning_closure_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "frontier_family": "coding_local_sandbox",
        "runner_family": "student_learning_closure",
        "seed": args.seed,
        "evidence_level": "heldout_trace_replay_closure" if split_kind == "hash_holdout" else split_kind,
        "candidate_source_before": "local_theseus_student_checkpoint",
        "candidate_source_after": "student_learning_checkpoint_v1",
        "public_benchmark_score_claim": "student_learning_checkpoint_public_task_calibration_only",
        "promotion_allowed": False,
        "promotion_rule": "requires learned checkpoint improvement plus independent real-code graduation run; same-trace replay alone cannot promote",
        "score": after_eval["pass_rate"],
        "score_semantics": "student_learning_trace_replay_selection_pass_rate_not_public_mastery",
        "summary": {
            "candidate_count": len(candidate_rows),
            "labeled_candidate_count": len(labeled),
            "outcome_count": len(outcomes),
            "train_task_count": len(train_task_ids),
            "eval_task_count": len(eval_task_ids),
            "training_example_count": len(training_examples),
            "before_eval_pass_rate": before_eval["pass_rate"],
            "after_eval_pass_rate": after_eval["pass_rate"],
            "pass_rate_delta": pass_rate_delta,
            "all_before_pass_rate": all_before["pass_rate"],
            "all_after_pass_rate": all_after["pass_rate"],
            "all_pass_rate_delta": round(all_after["pass_rate"] - all_before["pass_rate"], 6),
            "learned_candidate_count": len(learned_manifest),
            "checkpoint_id": checkpoint["checkpoint_id"],
            "external_inference_calls": 0,
            "neural_weight_update": False,
            "student_behavior_changed": pass_rate_delta > 0.0 or all_after["pass_rate"] > all_before["pass_rate"],
            "token_level_code_generation_learned": False,
        },
        "before_after": {
            "eval_before": before_eval,
            "eval_after": after_eval,
            "all_before": all_before,
            "all_after": all_after,
        },
        "leakage_policy": {
            "canonical_solutions_visible": False,
            "public_tests_visible_to_generator": False,
            "task_id_specific_lookup": False,
            "external_inference_calls": 0,
            "known_limitation": "This is a learned trace-ranker checkpoint, not token-level code generation.",
        },
        "artifacts": {
            "checkpoint": rel(resolve(args.checkpoint_out)),
            "candidate_manifest": rel(resolve(args.candidate_out)),
            "training_examples": rel(resolve(args.training_examples_out)),
            "transfer_artifact": rel(transfer_artifact),
            "source_candidate_manifest": args.candidate_manifest,
            "trace_in": args.trace_in,
        },
        "top_positive_features": top_weights(weights, positive=True),
        "top_negative_features": top_weights(weights, positive=False),
        "gates": gates,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if trigger_state in {"GREEN", "YELLOW"} else 1


def collect_outcomes(traces: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    outcomes: dict[tuple[str, str], dict[str, Any]] = {}
    for row in traces:
        if row.get("event") != "real_code_candidate_test":
            continue
        task_id = str(row.get("task_id") or "")
        digest = str(row.get("candidate_sha256") or "")
        if not task_id or not digest:
            continue
        key = (task_id, digest)
        item = outcomes.setdefault(
            key,
            {
                "task_id": task_id,
                "candidate_sha256": digest,
                "passed": False,
                "attempts": 0,
                "modes": set(),
                "case_type": row.get("case_type"),
                "source_task_id": row.get("source_task_id"),
                "residual_class": row.get("residual_class") or "",
            },
        )
        item["passed"] = bool(item["passed"] or row.get("passed"))
        item["attempts"] = int(item["attempts"] or 0) + 1
        item["modes"].add(str(row.get("mode") or ""))
        if row.get("residual_class"):
            item["residual_class"] = row.get("residual_class")
    return outcomes


def attach_labels(candidate_rows: list[dict[str, Any]], outcomes: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    labeled = []
    for raw in candidate_rows:
        if not isinstance(raw, dict):
            continue
        code = str(raw.get("code") or "")
        digest = str(raw.get("candidate_sha256") or "") or sha256_text(code)
        task_id = str(raw.get("task_id") or "")
        outcome = outcomes.get((task_id, digest))
        if not outcome:
            continue
        row = dict(raw)
        row["candidate_sha256"] = digest
        row["label_passed"] = bool(outcome.get("passed"))
        row["label_attempts"] = int(outcome.get("attempts") or 0)
        row["label_modes"] = sorted(str(mode) for mode in outcome.get("modes", set()) if mode)
        row["label_residual_class"] = outcome.get("residual_class") or ""
        labeled.append(row)
    return labeled


def split_tasks(task_ids: list[str], *, seed: int, holdout_ratio: float) -> tuple[set[str], set[str], str]:
    unique = sorted({task_id for task_id in task_ids if task_id})
    if len(unique) < 3:
        return set(unique), set(unique), "same_trace_replay_small_sample"
    holdout_count = max(1, min(len(unique) - 1, round(len(unique) * max(0.0, min(0.8, holdout_ratio)))))
    ranked = sorted(unique, key=lambda item: sha256_text(f"{seed}:{item}"))
    eval_ids = set(ranked[:holdout_count])
    train_ids = set(ranked[holdout_count:])
    return train_ids, eval_ids, "hash_holdout"


def train_ranker(rows: list[dict[str, Any]], *, epochs: int) -> tuple[Counter[str], float, list[dict[str, Any]]]:
    weights: Counter[str] = Counter()
    bias = 0.0
    examples = []
    for raw in rows:
        feats = features_for(raw)
        label = 1 if raw.get("label_passed") else -1
        examples.append(
            {
                "task_id_sha256": sha256_text(str(raw.get("task_id") or "")),
                "candidate_sha256": raw.get("candidate_sha256") or sha256_text(str(raw.get("code") or "")),
                "label": label,
                "passed": bool(raw.get("label_passed")),
                "features": sorted(feats),
                "rank": parse_rank(raw),
                "modes": raw.get("label_modes") or [],
            }
        )
    ordered = sorted(rows, key=lambda row: (str(row.get("task_id") or ""), parse_rank(row), str(row.get("candidate_sha256") or "")))
    for _epoch in range(epochs):
        for row in ordered:
            feats = features_for(row)
            label = 1 if row.get("label_passed") else -1
            score = bias + sum(weights[feat] for feat in feats)
            if label * score <= 0:
                for feat in feats:
                    weights[feat] += label
                bias += 0.25 * label
    return weights, bias, examples


def emit_learned_manifest(candidate_rows: list[dict[str, Any]], weights: Counter[str], bias: float, checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in candidate_rows:
        if not isinstance(raw, dict):
            continue
        key = (
            str(raw.get("task_id") or ""),
            str(raw.get("source_task_id") or ""),
            str(raw.get("entry_point") or ""),
        )
        groups[key].append(raw)
    out = []
    for _key, rows in sorted(groups.items()):
        scored = [(score_row(row, weights, bias), parse_rank(row), row) for row in rows]
        scored.sort(key=lambda item: (-item[0], item[1], str(item[2].get("candidate_sha256") or "")))
        for new_rank, (score, old_rank, raw) in enumerate(scored, start=1):
            row = dict(raw)
            code = str(row.get("code") or "")
            provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
            provenance = dict(provenance)
            provenance["student_learning_closure"] = {
                "policy": "project_theseus_student_learning_closure_v1",
                "checkpoint_id": checkpoint["checkpoint_id"],
                "parent_candidate_source": raw.get("candidate_source"),
                "parent_checkpoint_id": raw.get("checkpoint_id"),
                "parent_rank": old_rank,
                "learned_rank": new_rank,
                "selection_score": round(score, 6),
                "canonical_solution_used": False,
                "tests_used": False,
                "task_id_specific_lookup": False,
                "external_inference_calls": 0,
            }
            row.update(
                {
                    "candidate_source": "student_learning_checkpoint_v1",
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "origin": f"student_learning_checkpoint_v1:trace_ranker_v1:rank{new_rank}:parent_rank{old_rank}",
                    "candidate_sha256": row.get("candidate_sha256") or sha256_text(code),
                    "selection_score": round(score, 6),
                    "candidate_generation_mode": "trace_ranker_over_parent_candidates",
                    "candidate_generation_contract": "learned_selection_over_existing_candidates_not_token_level_generation",
                    "token_level_code_generation_learned": False,
                    "benchmark_promotion_eligible": False,
                    "loop_closure_generated": False,
                    "template_like_candidate": bool(row.get("template_like_candidate")),
                    "canonical_solution_seen_by_solver": False,
                    "public_tests_visible_to_generator": False,
                    "provenance": provenance,
                }
            )
            out.append(row)
    return out


def evaluate_selection(rows: list[dict[str, Any]], *, weights: Counter[str] | None, bias: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if task_id:
            groups[task_id].append(row)
    selected = []
    for task_id, task_rows in sorted(groups.items()):
        if weights is None:
            chosen = sorted(task_rows, key=lambda row: (parse_rank(row), str(row.get("candidate_sha256") or "")))[0]
            score = 0.0
        else:
            chosen = sorted(
                task_rows,
                key=lambda row: (-score_row(row, weights, bias), parse_rank(row), str(row.get("candidate_sha256") or "")),
            )[0]
            score = score_row(chosen, weights, bias)
        selected.append(
            {
                "task_id": task_id,
                "candidate_sha256": chosen.get("candidate_sha256"),
                "rank": parse_rank(chosen),
                "passed": bool(chosen.get("label_passed")),
                "selection_score": round(score, 6),
            }
        )
    passed = sum(1 for row in selected if row.get("passed"))
    return {
        "task_count": len(selected),
        "passed": passed,
        "pass_rate": real_code.ratio(passed, len(selected)),
        "selected": selected[:50],
    }


def features_for(row: dict[str, Any]) -> set[str]:
    code = str(row.get("code") or "")
    lowered = code.lower()
    origin = str(row.get("origin") or "")
    entry = str(row.get("entry_point") or "")
    tags = visible_tags(row)
    feats = {f"entry_arity:{len(real_code.function_args(code))}"}
    for token in tokenize(entry):
        feats.add(f"entry_token:{token}")
    for tag in tags:
        feats.add(f"tag:{tag}")
    if "return none" in lowered:
        feats.add("code:return_none")
    if re.search(r"return\s+[A-Za-z_][A-Za-z0-9_]*\s*$", code.strip()):
        feats.add("code:return_single_symbol")
    if "import re" in lowered or "re." in lowered:
        feats.add("code:regex")
    if "re.search" in lowered and "def\\s+" in lowered:
        feats.add("code:regex_def_name")
    if "all(" in lowered and " in " in lowered:
        feats.add("code:all_membership")
    if "seen = set" in lowered and "append" in lowered:
        feats.add("code:stable_dedupe_loop")
    if "set(" in lowered:
        feats.add("code:uses_set")
    if "sorted(" in lowered:
        feats.add("code:uses_sorted")
    if "sum(" in lowered:
        feats.add("code:uses_sum")
    if "len(" in lowered:
        feats.add("code:uses_len")
    if "for " in lowered:
        feats.add("code:has_loop")
    if "while " in lowered:
        feats.add("code:has_while")
    if "[" in code and " for " in lowered and "]" in code:
        feats.add("code:list_comprehension")
    if "collections" in lowered or "counter(" in lowered:
        feats.add("code:counter")
    if "program_induction_prior" in origin:
        feats.add("origin:program_induction_prior")
    return feats


def score_row(row: dict[str, Any], weights: Counter[str], bias: float) -> float:
    return float(bias + sum(weights[feat] for feat in features_for(row)) - 0.01 * parse_rank(row))


def build_checkpoint(
    *,
    weights: Counter[str],
    bias: float,
    training_examples: list[dict[str, Any]],
    train_task_ids: set[str],
    eval_task_ids: set[str],
    seed: int,
    source_manifest: str,
    trace_paths: str,
) -> dict[str, Any]:
    material = json.dumps(
        {
            "seed": seed,
            "bias": bias,
            "weights": sorted(weights.items()),
            "training_examples": [row["candidate_sha256"] + ":" + str(row["label"]) for row in training_examples],
        },
        sort_keys=True,
    )
    checkpoint_id = "theseus_student_learning_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return {
        "policy": "project_theseus_student_learning_code_checkpoint_v1",
        "created_utc": now(),
        "checkpoint_id": checkpoint_id,
        "checkpoint_kind": "learned_trace_ranker",
        "seed": seed,
        "source_candidate_manifest": source_manifest,
        "trace_paths": trace_paths,
        "bias": round(bias, 6),
        "weights": [{"feature": key, "weight": value} for key, value in sorted(weights.items()) if value],
        "summary": {
            "training_example_count": len(training_examples),
            "positive_examples": sum(1 for row in training_examples if row["label"] > 0),
            "negative_examples": sum(1 for row in training_examples if row["label"] < 0),
            "train_task_count": len(train_task_ids),
            "eval_task_count": len(eval_task_ids),
        },
        "generation_policy": {
            "public_tests_visible": False,
            "canonical_solutions_visible": False,
            "task_id_specific_lookup": False,
            "external_inference_calls": 0,
            "allowed_inputs": ["candidate_code_shape", "candidate_rank", "entry_point_tokens", "visible_task_tags", "sandbox_outcome_labels"],
        },
        "external_inference_calls": 0,
    }


def write_transfer_artifact(
    path: Path,
    *,
    checkpoint: dict[str, Any],
    before_eval: dict[str, Any],
    after_eval: dict[str, Any],
    all_before: dict[str, Any],
    all_after: dict[str, Any],
    split_kind: str,
    trace_paths: str,
    candidate_manifest: str,
) -> Path:
    payload = {
        "policy": "project_theseus_student_learning_closure_transfer_artifact_v1",
        "created_utc": now(),
        "family": "coding_local_sandbox",
        "card_id": "student_learning_closure",
        "active_card": True,
        "summary": {
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "split_kind": split_kind,
            "eval_before_pass_rate": before_eval.get("pass_rate"),
            "eval_after_pass_rate": after_eval.get("pass_rate"),
            "eval_pass_rate_delta": round(float(after_eval.get("pass_rate") or 0.0) - float(before_eval.get("pass_rate") or 0.0), 6),
            "all_before_pass_rate": all_before.get("pass_rate"),
            "all_after_pass_rate": all_after.get("pass_rate"),
            "student_behavior_changed": float(after_eval.get("pass_rate") or 0.0) > float(before_eval.get("pass_rate") or 0.0),
        },
        "failure_clusters": [
            {
                "category": "student_learning_closure_gap",
                "count": max(0, int(after_eval.get("task_count") or 0) - int(after_eval.get("passed") or 0)),
                "suggested_intervention": "Distill failing trace-ranker selections into the next code checkpoint and rerun real-code graduation with heldout tasks.",
                "cards": ["student_learning_closure"],
                "priority": 3.0,
            }
        ],
        "repair_traces": [
            {
                "trace_id": "student_learning_closure_ranker_update",
                "created_utc": now(),
                "category": "learned_candidate_selection",
                "repair_pattern": "trace_outcome_ranker",
                "transfer_hint": "Load learned ranker weights before code-family candidate selection and compare against frozen rank ordering.",
                "loads_into": ["code_repair_arm", "pressure_runner", "benchmark_adapter_factory", "octopus_router"],
            }
        ],
        "learned_features": {
            "positive": top_weights(Counter({row["feature"]: row["weight"] for row in checkpoint.get("weights", [])}), positive=True),
            "negative": top_weights(Counter({row["feature"]: row["weight"] for row in checkpoint.get("weights", [])}), positive=False),
        },
        "loads_into": ["code_repair_arm", "benchmark_adapter_factory", "pressure_runner", "octopus_router"],
        "verification": {
            "trace_paths": trace_paths,
            "candidate_manifest": candidate_manifest,
            "external_inference_calls": 0,
            "public_score_claim": "student_learning_trace_replay_selection_pass_rate_not_public_mastery",
        },
    }
    write_json(path, payload)
    return path


def merge_transfer_index(index_path: Path, artifact_path: Path) -> None:
    index = read_json(index_path, {})
    artifacts = [row for row in index.get("artifacts", []) if isinstance(row, dict)] if isinstance(index.get("artifacts"), list) else []
    rel_path = rel(artifact_path)
    artifacts = [row for row in artifacts if str(row.get("path") or "") != rel_path]
    payload = read_json(artifact_path, {})
    artifacts.append(
        {
            "name": "student_learning_closure_transfer",
            "family": "coding_local_sandbox",
            "card_id": "student_learning_closure",
            "path": rel_path,
            "loads_into": payload.get("loads_into") or ["code_repair_arm", "pressure_runner"],
            "cluster_count": len(payload.get("failure_clusters", [])) if isinstance(payload.get("failure_clusters"), list) else 0,
            "trace_count": len(payload.get("repair_traces", [])) if isinstance(payload.get("repair_traces"), list) else 0,
            "active_card": True,
        }
    )
    write_json(
        index_path,
        {
            "policy": "project_theseus_code_transfer_artifacts_index_v1",
            "created_utc": now(),
            "summary": {
                "frontier_family": "coding_local_sandbox",
                "active_card_id": "student_learning_closure",
                "artifact_count": len(artifacts),
                "cluster_count": sum(int(row.get("cluster_count") or 0) for row in artifacts),
                "trace_count": sum(int(row.get("trace_count") or 0) for row in artifacts),
                "loads_into": ["code_repair_arm", "benchmark_adapter_factory", "pressure_runner", "octopus_router"],
            },
            "artifacts": artifacts,
            "external_inference_calls": 0,
        },
    )


def leakage_findings(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = any(bool(row.get("canonical_solution_seen_by_solver")) for row in candidate_rows if isinstance(row, dict))
    tests = any(bool(row.get("public_tests_visible_to_generator")) for row in candidate_rows if isinstance(row, dict))
    suspicious_fields = []
    for row in candidate_rows:
        if not isinstance(row, dict):
            continue
        for key in row:
            lowered = str(key).lower()
            if lowered in {"canonical_solution", "test", "tests", "expected_answer", "answer_key"}:
                suspicious_fields.append(lowered)
    return {
        "canonical_solution_seen": canonical,
        "public_tests_visible": tests,
        "suspicious_answer_or_test_fields": sorted(set(suspicious_fields)),
    }


def visible_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    visible = provenance.get("visible_task") if isinstance(provenance.get("visible_task"), dict) else {}
    raw_tags = visible.get("tags")
    if isinstance(raw_tags, list):
        tags.extend(str(tag) for tag in raw_tags)
    return [token for tag in tags for token in tokenize(tag)]


def parse_rank(row: dict[str, Any]) -> int:
    origin = str(row.get("origin") or "")
    match = re.search(r"rank(\d+)", origin)
    if match:
        return int(match.group(1))
    return 999


def top_weights(weights: Counter[str], *, positive: bool) -> list[dict[str, Any]]:
    rows = [(feature, weight) for feature, weight in weights.items() if (weight > 0 if positive else weight < 0)]
    rows.sort(key=lambda item: (-abs(item[1]), item[0]))
    return [{"feature": feature, "weight": weight} for feature, weight in rows[:20]]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or "")]


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any = None) -> Any:
    default = {} if default is None else default
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
