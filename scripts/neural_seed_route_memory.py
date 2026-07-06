#!/usr/bin/env python3
"""Route-memory helpers for the neural-seed token decoder.

These helpers build and apply private-training-only semantic plan memories.
They are selection aids, not learned full-body generation. Callers must keep
candidate integrity and no-cheat audits responsible for any promotion claim.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any

from neural_seed_code_proposer_comparator import dict_or_empty, get_path, row_text, stable_hash
from neural_seed_token_decoder_support import return_shape_for_task, semantic_plan_from_body


CONTRACT_FINGERPRINT_SCOPES = [
    "exact_contract",
    "family_roles_shape",
    "family_shape_constructs",
    "family_shape",
]

CONTRACT_FINGERPRINT_SCOPE_WEIGHTS = {
    "exact_contract": 1.0,
    "family_roles_shape": 0.96,
    "family_shape_constructs": 0.92,
    "family_shape": 0.86,
}

CONTRACT_FEATURE_WEIGHTS = {
    "type_family": 3.0,
    "role": 2.0,
    "return_shape": 2.0,
    "required_construct": 1.0,
    "skeleton_bias": 0.75,
    "visible_arg_count_hint": 0.75,
    "must_preserve_container_shape": 0.5,
}


def generate_learned_semantic_route_candidates(
    model: Any,
    source_rows: list[list[int]],
    target_vocab: dict[str, int],
    *,
    top_k: int,
    route_memory: dict[str, Any] | None = None,
    prototype_route_weight: float = 0.0,
    prototype_route_keep_rate: float = 1.0,
    dropout_salt: str = "",
    torch: Any,
    device: Any,
) -> list[list[dict[str, Any]]]:
    if top_k <= 0 or not hasattr(model, "semantic_plan_logits"):
        return [[] for _row in source_rows]
    inverse = {idx: tok for tok, idx in target_vocab.items()}
    plan_ids = sorted((idx, tok) for tok, idx in target_vocab.items() if tok.startswith("SLOT:PLAN_"))
    if not plan_ids:
        return [[] for _row in source_rows]
    src = torch.tensor(source_rows, dtype=torch.long, device=device)
    model.eval()
    all_rows = []
    with torch.no_grad():
        logits = model.semantic_plan_logits(src)
        route_scores = logits.clone()
        use_route_memory = bool(route_memory and prototype_route_weight != 0.0)
        if use_route_memory:
            contexts = model.encode_source(src)
            contexts = contexts / contexts.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            prototypes = route_memory.get("prototype_tensor")
            plan_ids_for_prototypes = route_memory.get("plan_ids")
            if prototypes is not None and plan_ids_for_prototypes:
                for row_idx in range(src.shape[0]):
                    if not deterministic_keep(f"{dropout_salt}:{row_idx}", prototype_route_keep_rate):
                        continue
                    similarities = contexts[row_idx : row_idx + 1] @ prototypes.t()
                    for proto_idx, plan_id in enumerate(plan_ids_for_prototypes):
                        route_scores[row_idx, int(plan_id)] = route_scores[row_idx, int(plan_id)] + (
                            float(prototype_route_weight) * similarities[0, proto_idx]
                        )
        probs = torch.softmax(route_scores, dim=-1)
        for row_idx in range(src.shape[0]):
            ranked = sorted(
                ((idx, inverse.get(idx, ""), float(probs[row_idx, idx])) for idx, _tok in plan_ids),
                key=lambda item: item[2],
                reverse=True,
            )[:top_k]
            task_rows = []
            for rank, (_idx, tok, prob) in enumerate(ranked, start=1):
                tokens = [tok, "<eos>"]
                task_rows.append(
                    {
                        "body": "",
                        "decoded_tokens": tokens,
                        "rank_score": round(math.log(max(prob, 1e-9)), 8),
                        "decoded_token_count": len(tokens),
                        "decoded_token_sha256": stable_hash(" ".join(tokens)),
                        "beam_source": "learned_internal_semantic_route",
                        "learned_route_plan": tok.removeprefix("SLOT:PLAN_"),
                        "learned_route_probability": round(prob, 8),
                        "learned_route_rank": rank,
                        "learned_route_strategy": "plan_head_plus_context_prototype_memory"
                        if use_route_memory and deterministic_keep(f"{dropout_salt}:{row_idx}", prototype_route_keep_rate)
                        else "plan_head",
                    }
                )
            all_rows.append(task_rows)
    return all_rows


def build_learned_plan_route_memory(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    target_vocab: dict[str, int],
    *,
    enabled: bool,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    if not enabled or not hasattr(model, "encode_source"):
        return {}
    inverse = {idx: tok for tok, idx in target_vocab.items()}
    src = torch.tensor(source_rows, dtype=torch.long, device=device)
    targets = torch.tensor(target_rows, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        contexts = model.encode_source(src)
        contexts = contexts / contexts.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        plan_targets = targets[:, 1] if targets.shape[1] > 1 else targets[:, 0]
        plan_ids = sorted(
            int(plan_id)
            for plan_id in plan_targets.detach().cpu().tolist()
            if str(inverse.get(int(plan_id), "")).startswith("SLOT:PLAN_")
        )
        unique_plan_ids = sorted(set(plan_ids))
        prototypes = []
        counts: dict[str, int] = {}
        for plan_id in unique_plan_ids:
            mask = plan_targets.eq(int(plan_id))
            if int(mask.sum().detach().cpu()) <= 0:
                continue
            proto = contexts[mask].mean(dim=0)
            proto = proto / proto.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            prototypes.append(proto)
            counts[str(inverse.get(plan_id, plan_id)).removeprefix("SLOT:PLAN_")] = int(mask.sum().detach().cpu())
        if not prototypes:
            return {}
        return {
            "plan_ids": unique_plan_ids,
            "plan_tokens": [str(inverse.get(plan_id, "")) for plan_id in unique_plan_ids],
            "prototype_tensor": torch.stack(prototypes, dim=0),
            "plan_counts": counts,
        }


def learned_plan_route_memory_summary(route_memory: dict[str, Any]) -> dict[str, Any]:
    counts = dict_or_empty(route_memory.get("plan_counts"))
    return {
        "enabled": bool(route_memory),
        "prototype_count": len(route_memory.get("plan_ids") or []),
        "training_plan_count_total": sum(int(value) for value in counts.values()),
        "top_plan_counts": dict(Counter(counts).most_common(12)),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
    }


def build_contract_fingerprint_route_memory(
    train_rows: list[dict[str, Any]],
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {}
    scoped_counts: dict[str, dict[str, Counter[str]]] = {
        scope: defaultdict(Counter) for scope in CONTRACT_FINGERPRINT_SCOPES
    }
    plan_counts: Counter[str] = Counter()
    for row in train_rows:
        plan = semantic_plan_from_body(str(row.get("solution_body") or ""))
        if not plan:
            continue
        keys = contract_fingerprint_route_keys(row)
        if not keys:
            continue
        plan_counts[plan] += 1
        for scope, key in keys:
            scoped_counts[scope][key][plan] += 1
    memory_scopes = {
        scope: {
            key: {
                "total": int(sum(counts.values())),
                "plan_counts": dict(counts.most_common()),
            }
            for key, counts in key_counts.items()
        }
        for scope, key_counts in scoped_counts.items()
        if key_counts
    }
    if not memory_scopes:
        return {}
    return {
        "scopes": memory_scopes,
        "scope_order": list(CONTRACT_FINGERPRINT_SCOPES),
        "scope_weights": dict(CONTRACT_FINGERPRINT_SCOPE_WEIGHTS),
        "plan_counts": dict(plan_counts.most_common()),
        "policy": "train_contract_fingerprint_to_private_plan_memory_v0",
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
    }


def generate_contract_fingerprint_route_candidates(
    eval_rows: list[dict[str, Any]],
    target_vocab: dict[str, int],
    route_memory: dict[str, Any],
    *,
    top_k: int,
    keep_rate: float = 1.0,
    dropout_salt: str = "",
) -> list[list[dict[str, Any]]]:
    if top_k <= 0 or not route_memory:
        return [[] for _row in eval_rows]
    scope_order = [str(item) for item in route_memory.get("scope_order") or CONTRACT_FINGERPRINT_SCOPES]
    scope_weights = dict_or_empty(route_memory.get("scope_weights"))
    memory_scopes = dict_or_empty(route_memory.get("scopes"))
    all_rows = []
    for row_idx, row in enumerate(eval_rows):
        task_key = str(row.get("task_id") or row.get("entry_point") or row_idx)
        if not deterministic_keep(f"{dropout_salt}:{task_key}", keep_rate):
            all_rows.append([])
            continue
        best_by_plan: dict[str, dict[str, Any]] = {}
        keys_by_scope = dict(contract_fingerprint_route_keys(row))
        for scope in scope_order:
            key = keys_by_scope.get(scope)
            if not key:
                continue
            scope_memory = dict_or_empty(memory_scopes.get(scope))
            memory_row = dict_or_empty(scope_memory.get(key))
            total = int(memory_row.get("total") or 0)
            plan_counts = dict_or_empty(memory_row.get("plan_counts"))
            if total <= 0 or not plan_counts:
                continue
            scope_weight = float(scope_weights.get(scope) or CONTRACT_FINGERPRINT_SCOPE_WEIGHTS.get(scope, 0.0) or 0.0)
            for plan, count_value in plan_counts.items():
                plan = str(plan)
                if f"SLOT:PLAN_{plan}" not in target_vocab:
                    continue
                count = int(count_value or 0)
                if count <= 0:
                    continue
                score = scope_weight * (count / max(1, total))
                previous = best_by_plan.get(plan)
                if previous is None or score > float(previous.get("score") or 0.0):
                    best_by_plan[plan] = {
                        "score": score,
                        "scope": scope,
                        "support": count,
                        "total": total,
                    }
        ranked = sorted(best_by_plan.items(), key=lambda item: (-float(item[1]["score"]), -int(item[1]["support"]), item[0]))[:top_k]
        return_shape = return_shape_for_task(row)
        task_rows = []
        for rank, (plan, route) in enumerate(ranked, start=1):
            tokens = [f"SLOT:PLAN_{plan}"]
            if return_shape and return_shape != "unknown":
                tokens.append(f"SLOT:RETURN_SHAPE_{return_shape.upper()}")
            tokens.append("<eos>")
            task_rows.append(
                {
                    "body": "",
                    "decoded_tokens": tokens,
                    "rank_score": round(float(route["score"]), 8),
                    "decoded_token_count": len(tokens),
                    "decoded_token_sha256": stable_hash(" ".join(tokens)),
                    "beam_source": "learned_internal_semantic_route",
                    "learned_route_plan": plan,
                    "learned_route_probability": round(float(route["score"]), 8),
                    "learned_route_rank": rank,
                    "learned_route_strategy": "contract_fingerprint_context_memory",
                    "learned_route_fingerprint_scope": route.get("scope"),
                    "learned_route_fingerprint_support": route.get("support"),
                    "learned_route_fingerprint_total": route.get("total"),
                }
            )
        all_rows.append(task_rows)
    return all_rows


def contract_fingerprint_route_memory_summary(route_memory: dict[str, Any]) -> dict[str, Any]:
    memory_scopes = dict_or_empty(route_memory.get("scopes"))
    scope_summaries = {}
    for scope, rows in memory_scopes.items():
        rows = dict_or_empty(rows)
        conflicts = 0
        top_plans: Counter[str] = Counter()
        for memory_row in rows.values():
            plan_counts = dict_or_empty(dict_or_empty(memory_row).get("plan_counts"))
            if len(plan_counts) > 1:
                conflicts += 1
            for plan, count in plan_counts.items():
                top_plans[str(plan)] += int(count or 0)
        scope_summaries[str(scope)] = {
            "fingerprint_count": len(rows),
            "ambiguous_fingerprint_count": conflicts,
            "top_plan_counts": dict(top_plans.most_common(12)),
        }
    return {
        "enabled": bool(route_memory),
        "policy": route_memory.get("policy") or "train_contract_fingerprint_to_private_plan_memory_v0",
        "scope_summaries": scope_summaries,
        "training_plan_count_total": sum(int(value) for value in dict_or_empty(route_memory.get("plan_counts")).values()),
        "top_plan_counts": dict(Counter(dict_or_empty(route_memory.get("plan_counts"))).most_common(12)),
        "uses_only_allowed_decoder_contract_fields": True,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
    }


def contract_fingerprint_route_keys(row: dict[str, Any]) -> list[tuple[str, str]]:
    contract = dict_or_empty(row.get("decoder_contract"))
    roles = dict_or_empty(contract.get("argument_roles"))
    return_shape = str(
        get_path(contract, ["return_contract", "shape"], "")
        or contract.get("return_shape")
        or ""
    ).strip().lower()
    values = {
        "data_role": str(roles.get("data") or "").strip(),
        "other_role": str(roles.get("other") or "").strip(),
        "type_family": str(contract.get("type_family") or "").strip(),
        "return_shape": return_shape,
        "required_constructs": sorted(str(item) for item in contract.get("required_constructs", []) or []),
        "skeleton_bias": sorted(
            str(item)
            for item in get_path(contract, ["generation_plan", "skeleton_bias"], []) or []
        ),
        "visible_arg_count_hint": str(contract.get("visible_arg_count_hint") or "").strip(),
        "must_preserve_container_shape": str(
            get_path(contract, ["return_contract", "must_preserve_container_shape"], "")
        ).strip().lower(),
    }
    if not values["type_family"] and not values["return_shape"]:
        return []

    def key(scope: str, payload: dict[str, Any]) -> tuple[str, str]:
        return scope, stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    return [
        key("exact_contract", values),
        key(
            "family_roles_shape",
            {
                "type_family": values["type_family"],
                "data_role": values["data_role"],
                "other_role": values["other_role"],
                "return_shape": values["return_shape"],
                "visible_arg_count_hint": values["visible_arg_count_hint"],
            },
        ),
        key(
            "family_shape_constructs",
            {
                "type_family": values["type_family"],
                "return_shape": values["return_shape"],
                "required_constructs": values["required_constructs"],
                "skeleton_bias": values["skeleton_bias"],
            },
        ),
        key(
            "family_shape",
            {
                "type_family": values["type_family"],
                "return_shape": values["return_shape"],
            },
        ),
    ]


def build_contract_feature_route_memory(
    train_rows: list[dict[str, Any]],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    if not enabled:
        return []
    memory = []
    for row in train_rows:
        plan = semantic_plan_from_body(str(row.get("solution_body") or ""))
        if not plan:
            continue
        counts = contract_feature_counts(row)
        norm = math.sqrt(sum(value * value for value in counts.values()))
        if not counts or norm <= 0.0:
            continue
        memory.append({"plan": plan, "counts": counts, "norm": norm})
    return memory


def generate_contract_feature_route_candidates(
    eval_rows: list[dict[str, Any]],
    target_vocab: dict[str, int],
    route_memory: list[dict[str, Any]],
    *,
    top_k: int,
    keep_rate: float = 1.0,
    dropout_salt: str = "",
) -> list[list[dict[str, Any]]]:
    if top_k <= 0 or not route_memory:
        return [[] for _row in eval_rows]
    all_rows = []
    for row_idx, row in enumerate(eval_rows):
        task_key = str(row.get("task_id") or row.get("entry_point") or row_idx)
        if not deterministic_keep(f"{dropout_salt}:{task_key}", keep_rate):
            all_rows.append([])
            continue
        query = contract_feature_counts(row)
        query_norm = math.sqrt(sum(value * value for value in query.values()))
        scores: defaultdict[str, float] = defaultdict(float)
        support: Counter[str] = Counter()
        if query and query_norm > 0.0:
            for memory_row in route_memory:
                plan = str(memory_row.get("plan") or "")
                if f"SLOT:PLAN_{plan}" not in target_vocab:
                    continue
                score = sparse_cosine(query, query_norm, memory_row["counts"], float(memory_row["norm"]))
                if score <= 0.0:
                    continue
                support[plan] += 1
                if score > scores[plan]:
                    scores[plan] = score
        ranked = sorted(scores.items(), key=lambda item: (-item[1], -support[item[0]], item[0]))[:top_k]
        return_shape = return_shape_for_task(row)
        task_rows = []
        for rank, (plan, score) in enumerate(ranked, start=1):
            tokens = [f"SLOT:PLAN_{plan}"]
            if return_shape and return_shape != "unknown":
                tokens.append(f"SLOT:RETURN_SHAPE_{return_shape.upper()}")
            tokens.append("<eos>")
            task_rows.append(
                {
                    "body": "",
                    "decoded_tokens": tokens,
                    "rank_score": round(float(score), 8),
                    "decoded_token_count": len(tokens),
                    "decoded_token_sha256": stable_hash(" ".join(tokens)),
                    "beam_source": "learned_internal_semantic_route",
                    "learned_route_plan": plan,
                    "learned_route_probability": round(float(score), 8),
                    "learned_route_rank": rank,
                    "learned_route_strategy": "contract_feature_context_memory",
                    "learned_route_feature_support": int(support[plan]),
                    "learned_route_feature_top_score": round(float(score), 8),
                }
            )
        all_rows.append(task_rows)
    return all_rows


def contract_feature_route_memory_summary(route_memory: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("plan") or "") for row in route_memory)
    return {
        "enabled": bool(route_memory),
        "policy": "train_allowed_contract_feature_sparse_plan_memory_v0",
        "prototype_count": len(route_memory),
        "unique_plan_count": len(counts),
        "top_plan_counts": dict(counts.most_common(12)),
        "feature_weights": dict(CONTRACT_FEATURE_WEIGHTS),
        "uses_only_allowed_decoder_contract_fields": True,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
    }


def contract_feature_counts(row: dict[str, Any]) -> Counter[str]:
    contract = dict_or_empty(row.get("decoder_contract"))
    roles = dict_or_empty(contract.get("argument_roles"))
    generation_plan = dict_or_empty(contract.get("generation_plan"))
    return_contract = dict_or_empty(contract.get("return_contract"))
    counts: Counter[str] = Counter()

    def add(name: str, value: Any, weight_key: str) -> None:
        if value in (None, "", []):
            return
        weight = float(CONTRACT_FEATURE_WEIGHTS.get(weight_key, 1.0))
        counts[f"{name}={value}"] += weight

    add("type_family", str(contract.get("type_family") or "").strip(), "type_family")
    add(
        "return_shape",
        str(return_contract.get("shape") or contract.get("return_shape") or "").strip().lower(),
        "return_shape",
    )
    add("visible_arg_count_hint", str(contract.get("visible_arg_count_hint") or "").strip(), "visible_arg_count_hint")
    add(
        "must_preserve_container_shape",
        str(return_contract.get("must_preserve_container_shape") or "").strip().lower(),
        "must_preserve_container_shape",
    )
    for role_name, role_value in sorted(roles.items()):
        add(f"role.{role_name}", str(role_value).strip(), "role")
    for item in sorted(str(value) for value in contract.get("required_constructs", []) or []):
        add("required_construct", item, "required_construct")
    for item in sorted(str(value) for value in generation_plan.get("skeleton_bias", []) or []):
        add("skeleton_bias", item, "skeleton_bias")
    return counts


def build_visible_text_plan_route_memory(
    train_rows: list[dict[str, Any]],
    text_fields: list[str],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    if not enabled:
        return []
    memory = []
    for row in train_rows:
        plan = semantic_plan_from_body(str(row.get("solution_body") or ""))
        if not plan:
            continue
        counts = text_feature_counts(row_text(row, text_fields))
        norm = math.sqrt(sum(value * value for value in counts.values()))
        if not counts or norm <= 0.0:
            continue
        memory.append({"plan": plan, "counts": counts, "norm": norm})
    return memory


def generate_visible_text_prototype_route_candidates(
    eval_rows: list[dict[str, Any]],
    text_fields: list[str],
    route_memory: list[dict[str, Any]],
    *,
    top_k: int,
    keep_rate: float = 1.0,
    dropout_salt: str = "",
) -> list[list[dict[str, Any]]]:
    if top_k <= 0 or not route_memory:
        return [[] for _row in eval_rows]
    all_rows = []
    for row_idx, row in enumerate(eval_rows):
        task_key = str(row.get("task_id") or row.get("entry_point") or row_idx)
        if not deterministic_keep(f"{dropout_salt}:{task_key}", keep_rate):
            all_rows.append([])
            continue
        query = text_feature_counts(row_text(row, text_fields))
        query_norm = math.sqrt(sum(value * value for value in query.values()))
        scores: defaultdict[str, float] = defaultdict(float)
        if query and query_norm > 0.0:
            for memory_row in route_memory:
                score = sparse_cosine(query, query_norm, memory_row["counts"], float(memory_row["norm"]))
                if score > scores[str(memory_row["plan"])]:
                    scores[str(memory_row["plan"])] = score
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        task_rows = []
        for rank, (plan, score) in enumerate(ranked, start=1):
            tokens = [f"SLOT:PLAN_{plan}", "<eos>"]
            task_rows.append(
                {
                    "body": "",
                    "decoded_tokens": tokens,
                    "rank_score": round(float(score), 8),
                    "decoded_token_count": len(tokens),
                    "decoded_token_sha256": stable_hash(" ".join(tokens)),
                    "beam_source": "learned_internal_semantic_route",
                    "learned_route_plan": plan,
                    "learned_route_probability": round(float(score), 8),
                    "learned_route_rank": rank,
                    "learned_route_strategy": "visible_text_prototype_memory",
                }
            )
        all_rows.append(task_rows)
    return all_rows


def deterministic_keep(key: str, keep_rate: float) -> bool:
    if keep_rate >= 1.0:
        return True
    if keep_rate <= 0.0:
        return False
    digest = stable_hash(str(key))
    bucket = int(digest[:12], 16) / float(16**12)
    return bucket < keep_rate


def text_feature_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw in str(text or "").split():
        token = raw.strip().lower()
        if not token:
            continue
        counts[token] += 1
        for part in token.replace(":", " ").replace("_", " ").split():
            if part:
                counts[part] += 1
    return counts


def sparse_cosine(left: Counter[str], left_norm: float, right: Counter[str], right_norm: float) -> float:
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(float(value) * float(right.get(key, 0.0)) for key, value in left.items())
    return dot / max(1e-9, left_norm * right_norm)


def visible_text_plan_route_memory_summary(route_memory: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("plan") or "") for row in route_memory)
    return {
        "enabled": bool(route_memory),
        "prototype_count": len(route_memory),
        "unique_plan_count": len(counts),
        "top_plan_counts": dict(counts.most_common(12)),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
    }
