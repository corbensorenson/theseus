#!/usr/bin/env python3
"""Materialize licensed auxiliary objectives for canonical MoECOT arms.

This owner materializes code denoising and the KERC English objective views. It
does not train another model or grant capability credit to deterministic record
validation and compilation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from moecot_language_supervision import (
    BoundedRows,
    now,
    read_json,
    relative,
    resolve,
    sha256_file,
    write_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from moecot_language_tokenizer import exact_text_tokens
from neural_seed_open_vocab import (
    bound_logical_tokens,
    decode_target_tokens,
    encode_tokens,
    populate_open_vocab,
)
from kernel_english_protocol import (
    SEMANTIC_EVIDENCE_TIERS,
    SEMANTIC_SUPERVISION_POLICY,
    TRAINING_OBJECTIVES,
    TRAINING_VERIFICATION_POLICY,
    compile_training_views,
    kernel_training_contract,
    validate_training_disposition,
    validate_training_record,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moecot_language_arm_training.json"
ARM_IDS = ("english", "python", "javascript_typescript", "html_css", "rust")
KERC_KERNEL_OBJECTIVES = {
    "surface_to_kernel_program_v1",
    "kernel_program_to_answer_packet_v1",
}
KERC_POINTER_TOKEN_RE = re.compile(
    r"(?:@[A-Z][A-Za-z0-9_]*|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\Z"
)
KERC_POINTER_CONTROL_TOKENS = {
    "{",
    "}",
    "[",
    "]",
    "(",
    ")",
    ":",
    ",",
    '"',
    "\\",
    " ",
    "\n",
    "\r",
    "\t",
}
KERC_SOURCE_CATALOG_POLICY = "project_theseus_kerc_semantic_source_catalog_v1"
KERC_SEMANTIC_PROGRAM_POLICY = "project_theseus_kerc_semantic_supervision_program_v1"
KERC_SEMANTIC_CORPUS_POLICY = "project_theseus_kerc_semantic_corpus_materialization_v1"


def kerc_code_tokens(text: str) -> list[str]:
    """Losslessly tokenize typed Kernel/answer JSON while preserving handles."""

    raw = bound_logical_tokens(exact_text_tokens(text))
    tokens: list[str] = []
    index = 0
    while index < len(raw):
        if (
            raw[index] == "@"
            and index + 1 < len(raw)
            and str(raw[index + 1]).replace("_", "").isalnum()
        ):
            tokens.append("@" + str(raw[index + 1]))
            index += 2
            continue
        tokens.append(str(raw[index]))
        index += 1
    if "".join(tokens) != str(text):
        raise ValueError("KERC code tokenizer failed exact reconstruction")
    return tokens


def kerc_surface_tokens(text: str) -> list[str]:
    """Tokenize arbitrary KERC surface text without oversized unknown atoms."""

    tokens = bound_logical_tokens(exact_text_tokens(text))
    if "".join(tokens) != str(text):
        raise ValueError("KERC surface tokenizer failed exact reconstruction")
    return tokens


def kerc_code_space(token: str) -> str:
    value = str(token)
    if (
        value in KERC_POINTER_CONTROL_TOKENS
        or value.isspace()
        or KERC_POINTER_TOKEN_RE.fullmatch(value)
    ):
        return "V_P"
    return "V_K"


def build_kerc_code_vocabulary(
    private_train_views: list[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    """Fit V_K/V_P on private-train positive targets only."""

    kernel_counts: Counter[str] = Counter()
    pointer_counts: Counter[str] = Counter()
    source_view_count = 0
    for view in private_train_views:
        if str(view.get("objective") or "") not in KERC_KERNEL_OBJECTIVES:
            continue
        source_view_count += 1
        for token in kerc_code_tokens(str(view.get("target") or "")):
            (pointer_counts if kerc_code_space(token) == "V_P" else kernel_counts)[
                token
            ] += 1
    if not source_view_count or not kernel_counts or not pointer_counts:
        raise ValueError("KERC code vocabulary requires compiler/core private-train views")
    kernel_vocab = {"<pad>": 0, "<unk>": 1}
    pointer_vocab = {"<pad>": 0, "<unk>": 1}
    populate_open_vocab(
        kernel_vocab,
        kernel_counts,
        max_vocab=int(contract["kernel_max_vocab"]),
        stream="target",
    )
    populate_open_vocab(
        pointer_vocab,
        pointer_counts,
        max_vocab=int(contract["pointer_max_vocab"]),
        stream="target",
    )
    payload = {
        "policy": "project_theseus_kerc_dual_code_vocabulary_v1",
        "fit_split": "private_train",
        "fit_positive_targets_only": True,
        "dev_eval_vocabulary_fit_count": 0,
        "verifier_corruption_vocabulary_fit_count": 0,
        "surface_vocabulary_owner": "canonical_moecot_target_vocab",
        "kernel_max_vocab": int(contract["kernel_max_vocab"]),
        "pointer_max_vocab": int(contract["pointer_max_vocab"]),
        "kernel_vocab": kernel_vocab,
        "pointer_vocab": pointer_vocab,
        "kernel_observed_token_count": int(sum(kernel_counts.values())),
        "pointer_observed_token_count": int(sum(pointer_counts.values())),
        "source_view_count": source_view_count,
        "tokenizer": "lossless_exact_json_with_typed_handle_coalescing_v1",
        "byte_fallback_required": True,
        "fallback_return_count": 0,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
    }
    payload["contract_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def encode_kerc_view_target(
    view: dict[str, Any],
    *,
    target_vocab: dict[str, int],
    code_vocabulary: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    objective = str(view.get("objective") or "")
    target = str(view.get("target") or "")
    if objective not in KERC_KERNEL_OBJECTIVES:
        return encode_tokens(kerc_surface_tokens(target), target_vocab, stream="target")
    ids: list[int] = []
    unknown = 0
    fallback_tokens = 0
    by_space = {"V_K": 0, "V_P": 0}
    kernel_vocab = code_vocabulary.get("kernel_vocab") or {}
    pointer_vocab = code_vocabulary.get("pointer_vocab") or {}
    for token in kerc_code_tokens(target):
        space = kerc_code_space(token)
        vocab = pointer_vocab if space == "V_P" else kernel_vocab
        encoded, receipt = encode_tokens([token], vocab, stream="target")
        ids.extend(encoded)
        unknown += int(receipt.get("unknown_token_count") or 0)
        fallback_tokens += int(receipt.get("fallback_token_count") or 0)
        by_space[space] += len(encoded)
    return ids, {
        "policy": "project_theseus_kerc_dual_code_encoding_v1",
        "unknown_token_count": unknown,
        "fallback_token_count": fallback_tokens,
        "encoded_token_count": len(ids),
        "encoded_tokens_by_space": by_space,
        "code_vocabulary_sha256": code_vocabulary.get("contract_sha256"),
        "failure_behavior": "reject_without_surface_or_template_fallback",
    }


def encode_kerc_global_target(
    text: str,
    *,
    code_vocabulary: dict[str, Any],
    kernel_offset: int,
    pointer_offset: int,
) -> tuple[list[int], dict[str, Any]]:
    """Encode a Kernel/answer target into disjoint global V_K/V_P ranges."""

    ids: list[int] = []
    unknown = 0
    fallback_tokens = 0
    by_space = {"V_K": 0, "V_P": 0}
    kernel_vocab = code_vocabulary.get("kernel_vocab") or {}
    pointer_vocab = code_vocabulary.get("pointer_vocab") or {}
    for token in kerc_code_tokens(text):
        space = kerc_code_space(token)
        vocab = pointer_vocab if space == "V_P" else kernel_vocab
        offset = pointer_offset if space == "V_P" else kernel_offset
        encoded, receipt = encode_tokens([token], vocab, stream="target")
        ids.extend(offset + int(value) for value in encoded)
        unknown += int(receipt.get("unknown_token_count") or 0)
        fallback_tokens += int(receipt.get("fallback_token_count") or 0)
        by_space[space] += len(encoded)
    return ids, {
        "policy": "project_theseus_kerc_global_dual_code_encoding_v1",
        "unknown_token_count": unknown,
        "fallback_token_count": fallback_tokens,
        "encoded_token_count": len(ids),
        "encoded_tokens_by_space": by_space,
        "kernel_offset": int(kernel_offset),
        "pointer_offset": int(pointer_offset),
        "code_vocabulary_sha256": code_vocabulary.get("contract_sha256"),
        "failure_behavior": "reject_without_surface_or_template_fallback",
    }


def decode_kerc_global_target(
    ids: list[int],
    *,
    code_vocabulary: dict[str, Any],
    kernel_offset: int,
    pointer_offset: int,
) -> tuple[str, dict[str, Any]]:
    kernel_inverse = {
        int(value): str(token)
        for token, value in (code_vocabulary.get("kernel_vocab") or {}).items()
    }
    pointer_inverse = {
        int(value): str(token)
        for token, value in (code_vocabulary.get("pointer_vocab") or {}).items()
    }
    logical: list[str] = []
    by_space = {"V_K": 0, "V_P": 0}
    for global_id in ids:
        value = int(global_id)
        if kernel_offset <= value < pointer_offset:
            token = kernel_inverse.get(value - kernel_offset)
            space = "V_K"
        elif value >= pointer_offset:
            token = pointer_inverse.get(value - pointer_offset)
            space = "V_P"
        else:
            token = None
            space = ""
        if token is None:
            return "", {
                "policy": "project_theseus_kerc_global_dual_code_decoding_v1",
                "state": "FAULT",
                "reason": "unassigned_or_cross_space_token",
                "token_id": value,
                "failure_behavior": "reject_without_surface_or_template_fallback",
            }
        logical.append(token)
        by_space[space] += 1
    decoded, receipt = decode_target_tokens(logical)
    if receipt.get("state") != "READY":
        return "", {
            "policy": "project_theseus_kerc_global_dual_code_decoding_v1",
            "state": "FAULT",
            "reason": "byte_fallback_decode_fault",
            "open_vocab": receipt,
            "failure_behavior": "reject_without_surface_or_template_fallback",
        }
    return "".join(decoded), {
        "policy": "project_theseus_kerc_global_dual_code_decoding_v1",
        "state": "READY",
        "decoded_tokens_by_space": by_space,
        "exact_reconstruction": True,
        "fallback_return_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kernel-english", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    cfg = (
        validate_kernel_english_config(config)
        if args.kernel_english
        else validate_config(config)
    )
    if args.kernel_english:
        report = (
            materialize_kernel_english(config, config_path)
            if args.execute
            else inspect_kernel_english(config, config_path)
        )
    else:
        report = materialize(config, config_path) if args.execute else inspect(config, config_path)
    write_json(resolve(args.out or cfg["report"]), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PLANNED"} else 2


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("source_conditioned_pretraining")
    cfg = cfg if isinstance(cfg, dict) else {}
    if cfg.get("policy") != "project_theseus_moecot_source_conditioned_pretraining_v1":
        raise ValueError("unexpected source-conditioned pretraining policy")
    if tuple((cfg.get("rows_by_arm") or {}).keys()) != ARM_IDS:
        raise ValueError("source-conditioned row arm set/order mismatch")
    if int((cfg.get("rows_by_arm") or {}).get("english") or 0) != 0:
        raise ValueError("code-denoising source cannot be assigned to the English arm")
    if not 0.0 < float(cfg.get("deletion_fraction") or 0.0) < 0.5:
        raise ValueError("deletion fraction must be bounded between zero and one half")
    if int(cfg.get("maximum_windows_per_document") or 0) <= 0:
        raise ValueError("maximum windows per document must be positive")
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int(cfg.get(key) or 0):
            raise ValueError(f"source-conditioned no-cheat counter must remain zero: {key}")
    return cfg


def validate_kernel_english_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("kernel_english_training")
    cfg = cfg if isinstance(cfg, dict) else {}
    if cfg.get("policy") != "project_theseus_moecot_kernel_english_stage_v1":
        raise ValueError("unexpected KERC training-stage policy")
    disposition = validate_training_disposition(cfg)
    full_kerc_enabled = disposition.get("full_kerc_training_enabled") is True
    if tuple(cfg.get("objective_order") or ()) != TRAINING_OBJECTIVES:
        raise ValueError("KERC objective order/identity mismatch")
    rows = cfg.get("records_by_split") or {}
    if tuple(rows) != ("private_train", "private_dev", "private_eval"):
        raise ValueError("KERC record split set/order mismatch")
    if full_kerc_enabled:
        if any(int(value or 0) <= 0 for value in rows.values()):
            raise ValueError("KERC record floors must be positive for every split")
        if not cfg.get("allowed_licenses"):
            raise ValueError("KERC stage requires an explicit license allowlist")
        if not str(cfg.get("verification_ledger_jsonl") or "").strip():
            raise ValueError("KERC stage requires a separate verification ledger")
        if not str(cfg.get("semantic_source_catalog_json") or "").strip():
            raise ValueError("KERC stage requires a semantic source catalog")
        validate_kerc_semantic_program(cfg)
        validate_kerc_semantic_corpus_config(cfg)
    elif any(int(value or 0) != 0 for value in rows.values()):
        raise ValueError("retired KERC stage must request zero records")
    if int(cfg.get("maximum_sequence_tokens") or 0) <= 0:
        raise ValueError("KERC maximum sequence tokens must be positive")
    if not 1 <= int(cfg.get("batch_size") or 0) <= 16:
        raise ValueError("KERC batch size must be bounded")
    vocabulary = cfg.get("code_vocabulary") or {}
    if (
        vocabulary.get("policy") != "project_theseus_kerc_dual_code_vocabulary_v1"
        or vocabulary.get("fit_split") != "private_train"
        or vocabulary.get("surface_vocabulary_owner")
        != "canonical_moecot_target_vocab"
        or vocabulary.get("byte_fallback_required") is not True
        or vocabulary.get("dev_eval_vocabulary_fit_forbidden") is not True
        or int(vocabulary.get("kernel_max_vocab") or 0) < 512
        or int(vocabulary.get("pointer_max_vocab") or 0) < 512
    ):
        raise ValueError("KERC dual-code vocabulary contract is incomplete")
    for key in (
        "public_training_rows_written",
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
        "template_credit",
        "deterministic_renderer_credit",
        "candidate_generation_credit",
    ):
        if int(cfg.get(key) or 0):
            raise ValueError(f"KERC no-cheat counter must remain zero: {key}")
    return cfg


def validate_kerc_semantic_program(cfg: dict[str, Any]) -> dict[str, Any]:
    program = cfg.get("semantic_supervision")
    program = program if isinstance(program, dict) else {}
    if program.get("policy") != KERC_SEMANTIC_PROGRAM_POLICY:
        raise ValueError("KERC semantic-supervision program policy mismatch")
    tiers = program.get("tiers") if isinstance(program.get("tiers"), dict) else {}
    if tuple(tiers) != tuple(SEMANTIC_EVIDENCE_TIERS):
        raise ValueError("KERC semantic evidence tier set/order mismatch")
    for tier, contract in SEMANTIC_EVIDENCE_TIERS.items():
        configured = tiers.get(tier) if isinstance(tiers.get(tier), dict) else {}
        expected = {
            "claim_authority": contract["claim_authority"],
            "maximum_optimizer_sampling_weight": float(
                contract["maximum_optimizer_sampling_weight"]
            ),
            "training_only": contract["allowed_splits"] == {"private_train"},
        }
        if configured != expected:
            raise ValueError(f"KERC semantic evidence tier contract mismatch: {tier}")
    floors = program.get("minimum_decision_grade_records_by_split_and_objective") or {}
    requested = cfg.get("records_by_split") or {}
    if tuple(floors) != tuple(requested):
        raise ValueError("KERC decision-grade split floor set/order mismatch")
    for split, objective_floors in floors.items():
        if (
            not isinstance(objective_floors, dict)
            or tuple(objective_floors) != TRAINING_OBJECTIVES
        ):
            raise ValueError(f"KERC decision-grade objective floor set/order invalid: {split}")
        for objective, floor in objective_floors.items():
            if not 0 <= int(floor) <= int(requested[split]):
                raise ValueError(
                    f"KERC decision-grade objective floor invalid: {split}:{objective}"
                )
    record_caps = program.get("maximum_train_record_share_by_tier") or {}
    probability_caps = program.get("maximum_train_optimizer_probability_by_tier") or {}
    if set(record_caps) != {"local_parser_silver", "governed_openai_residual"}:
        raise ValueError("KERC train record-share caps are incomplete")
    if set(probability_caps) != {"governed_openai_residual"}:
        raise ValueError("KERC optimizer-probability cap is incomplete")
    if not 0.0 <= float(record_caps["local_parser_silver"]) <= 0.9:
        raise ValueError("KERC parser-silver record share may not exceed 0.9")
    if not 0.0 <= float(record_caps["governed_openai_residual"]) <= 0.1:
        raise ValueError("KERC teacher residual record share may not exceed 0.1")
    if not 0.0 <= float(probability_caps["governed_openai_residual"]) <= 0.02:
        raise ValueError("KERC teacher residual optimizer probability may not exceed 0.02")
    if program.get("public_semantic_benchmarks_training_forbidden") is not True:
        raise ValueError("KERC public semantic benchmarks must remain calibration-only")
    if program.get("silver_can_satisfy_decision_grade_floor") is not False:
        raise ValueError("KERC silver rows may not satisfy decision-grade floors")
    qualifications = program.get("source_qualification")
    if not isinstance(qualifications, list) or not qualifications:
        raise ValueError("KERC semantic source qualification ledger is required")
    identities: set[str] = set()
    for row in qualifications:
        if not isinstance(row, dict):
            raise ValueError("KERC semantic source qualification row is invalid")
        source_id = str(row.get("source_id") or "")
        disposition = str(row.get("disposition") or "")
        if (
            not source_id
            or source_id in identities
            or str(row.get("intended_tier") or "") not in SEMANTIC_EVIDENCE_TIERS
            or not disposition
            or not str(row.get("license_spdx") or "")
            or not str(row.get("source_url") or "")
        ):
            raise ValueError(f"KERC semantic source qualification invalid: {source_id}")
        if row.get("public_benchmark_surface") is True and disposition.startswith(
            "eligible"
        ):
            raise ValueError(f"public semantic benchmark cannot be training-eligible: {source_id}")
        identities.add(source_id)
    return program


def validate_kerc_semantic_corpus_config(cfg: dict[str, Any]) -> dict[str, Any]:
    corpus = cfg.get("semantic_corpus_materialization")
    corpus = corpus if isinstance(corpus, dict) else {}
    if corpus.get("policy") != KERC_SEMANTIC_CORPUS_POLICY:
        raise ValueError("KERC semantic corpus materialization policy mismatch")
    sources = {name: corpus.get(name) or {} for name in ("dolly", "masc", "oasst2")}
    for name, source in sources.items():
        path_key = "archive_path" if name == "masc" else "path"
        source_path_ready = (
            isinstance(source.get("files"), dict)
            and tuple(source["files"]) == ("train", "validation")
            and all(
                str(row.get("path") or "")
                and re.fullmatch(
                    r"sha256:[0-9a-f]{64}", str(row.get("content_sha256") or "")
                )
                for row in source["files"].values()
            )
            if name == "oasst2"
            else bool(str(source.get(path_key) or ""))
        )
        if (
            not source_path_ready
            or not str(source.get("dataset_id") or "")
            or not str(source.get("dataset_revision") or "")
            or not str(source.get("source_url") or "").startswith("https://")
            or not str(source.get("license_evidence_url") or "").startswith("https://")
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(source.get("content_sha256") or ""))
            or not str(source.get("license_spdx") or "")
            or tuple(source.get("records_by_split") or {})
            != ("private_train", "private_dev", "private_eval")
            or any(int(value or 0) < 0 for value in (source.get("records_by_split") or {}).values())
            or not set(source.get("allowed_objectives") or {}) <= set(TRAINING_OBJECTIVES)
            or not source.get("allowed_objectives")
        ):
            raise ValueError(f"KERC semantic corpus source contract invalid: {name}")
    requested = cfg.get("records_by_split") or {}
    grounded_counts = sources["dolly"].get("grounded_question_records_by_split")
    grounded_objectives = sources["dolly"].get("grounded_question_allowed_objectives")
    grounded_forms = sources["dolly"].get("grounded_question_required_forms")
    if (
        not isinstance(grounded_counts, dict)
        or tuple(grounded_counts) != ("private_train", "private_dev", "private_eval")
        or any(int(value) < 0 for value in grounded_counts.values())
        or not isinstance(grounded_objectives, list)
        or set(grounded_objectives) != set(TRAINING_OBJECTIVES)
        or not isinstance(grounded_forms, list)
        or len(grounded_forms) < 4
        or len(set(grounded_forms)) != len(grounded_forms)
        or any(not str(value) for value in grounded_forms)
        or not str(sources["dolly"].get("grounded_question_claim_scope") or "")
    ):
        raise ValueError("KERC Dolly grounded-question contract invalid")
    behavior_counts = sources["oasst2"].get("explicit_behavior_records_by_split")
    if (
        not isinstance(behavior_counts, dict)
        or tuple(behavior_counts) != ("private_train", "private_dev", "private_eval")
        or any(
            set((behavior_counts.get(split) or {})) != {"CLARIFY", "ABSTAIN"}
            or any(int(value) < 0 for value in behavior_counts[split].values())
            for split in behavior_counts
        )
        or not str(sources["oasst2"].get("explicit_behavior_claim_scope") or "")
    ):
        raise ValueError("KERC OASST2 explicit behavior contract invalid")
    for split in requested:
        total = sum(int(source["records_by_split"][split]) for source in sources.values())
        total += int(grounded_counts[split])
        total += sum(int(value) for value in behavior_counts[split].values())
        if total != int(requested[split]):
            raise ValueError(f"KERC semantic corpus split total mismatch: {split}")
    floors = cfg["semantic_supervision"][
        "minimum_decision_grade_records_by_split_and_objective"
    ]
    for split, objective_floors in floors.items():
        for objective, floor in objective_floors.items():
            available = sum(
                int(source["records_by_split"][split])
                for source in sources.values()
                if objective in source["allowed_objectives"]
            )
            if objective in grounded_objectives:
                available += int(grounded_counts[split])
            if objective in sources["oasst2"]["allowed_objectives"]:
                available += sum(int(value) for value in behavior_counts[split].values())
            if available < int(floor):
                raise ValueError(
                    f"KERC semantic corpus cannot satisfy objective floor: {split}:{objective}"
                )
    groups = sources["masc"].get("document_groups") or {}
    if tuple(groups) != ("private_dev", "private_eval"):
        raise ValueError("KERC MASC heldout document groups are incomplete")
    dev = {str(value) for value in groups["private_dev"]}
    evaluation = {str(value) for value in groups["private_eval"]}
    if not dev or not evaluation or dev & evaluation:
        raise ValueError("KERC MASC heldout document groups overlap or are empty")
    oasst = sources["oasst2"]
    if (
        oasst.get("required_valid_realization_ranks") != [0, 1]
        or not 0.0 <= float(oasst.get("minimum_quality", -1.0)) <= 1.0
        or set(oasst.get("maximum_label_values") or {})
        != {"spam", "lang_mismatch", "pii", "not_appropriate"}
        or any(
            not 0.0 <= float(value) <= 1.0
            for value in (oasst.get("maximum_label_values") or {}).values()
        )
        or any(
            int(oasst.get(key) or 0) <= 0
            for key in (
                "maximum_current_characters",
                "maximum_response_characters",
                "maximum_context_characters",
                "maximum_compiled_context_bytes",
                "minimum_prior_turns",
                "maximum_prior_turns",
            )
        )
        or int(oasst.get("minimum_prior_turns") or 0)
        > int(oasst.get("maximum_prior_turns") or 0)
    ):
        raise ValueError("KERC OASST2 conversation-tree contract is incomplete")
    for key in ("minimum_source_groups_by_split", "minimum_source_sentences_by_split"):
        values = corpus.get(key) or {}
        if tuple(values) != ("private_train", "private_dev", "private_eval") or any(
            int(value or 0) <= 0 for value in values.values()
        ):
            raise ValueError(f"KERC semantic corpus diversity floor invalid: {key}")
    if int(corpus.get("maximum_source_characters") or 0) < 256:
        raise ValueError("KERC semantic corpus source-character cap is too small")
    for key in (
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
        "template_credit",
    ):
        if int(corpus.get(key) or 0):
            raise ValueError(f"KERC semantic corpus no-cheat counter must remain zero: {key}")
    return corpus


def load_kerc_semantic_source_catalog(
    path: Path, cfg: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payload = read_json(path) if path.is_file() else {}
    gaps: list[str] = []
    if payload.get("policy") != KERC_SOURCE_CATALOG_POLICY:
        return {}, ["kernel_semantic_source_catalog_policy_invalid"]
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        return {}, ["kernel_semantic_source_catalog_empty"]
    sources: dict[str, dict[str, Any]] = {}
    allowed_licenses = {str(value).lower() for value in cfg.get("allowed_licenses") or []}
    for index, row in enumerate(raw_sources):
        if not isinstance(row, dict):
            gaps.append(f"kernel_semantic_source_catalog_row_invalid:{index}")
            continue
        dataset_id = str(row.get("dataset_id") or "")
        if not dataset_id or dataset_id in sources:
            gaps.append(f"kernel_semantic_source_catalog_identity_invalid:{index}")
            continue
        required = (
            str(row.get("dataset_revision") or "").strip()
            and re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("content_sha256") or ""))
            and str(row.get("license_spdx") or "").lower() in allowed_licenses
            and row.get("permitted_use") == "model_training"
            and row.get("training_allowed") is True
            and row.get("public_benchmark_surface") is False
            and row.get("public_benchmark_payload") is False
        )
        tiers = row.get("allowed_evidence_tiers")
        objectives = row.get("allowed_objectives")
        if (
            not required
            or not isinstance(tiers, list)
            or not tiers
            or any(str(tier) not in SEMANTIC_EVIDENCE_TIERS for tier in tiers)
            or not isinstance(objectives, list)
            or not objectives
            or any(str(objective) not in TRAINING_OBJECTIVES for objective in objectives)
        ):
            gaps.append(f"kernel_semantic_source_catalog_contract_invalid:{dataset_id}")
            continue
        sources[dataset_id] = row
    return sources, sorted(set(gaps))


def validate_kerc_record_source(
    record: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> str:
    provenance = record.get("provenance") or {}
    dataset_id = str(provenance.get("dataset_id") or "")
    source = sources.get(dataset_id)
    if source is None:
        return "semantic_source_absent_from_catalog"
    semantic = record.get("semantic_supervision") or {}
    authorized_objectives = {
        objective
        for objective, authorized in (semantic.get("objective_authority") or {}).items()
        if authorized is True
    }
    checks = (
        str(provenance.get("dataset_revision") or "") == str(source.get("dataset_revision") or ""),
        str(provenance.get("license_spdx") or "").lower()
        == str(source.get("license_spdx") or "").lower(),
        str(semantic.get("evidence_tier") or "")
        in {str(value) for value in source.get("allowed_evidence_tiers") or []},
        str(semantic.get("annotation_source_sha256") or "")
        == str(source.get("content_sha256") or ""),
        authorized_objectives
        <= {str(value) for value in source.get("allowed_objectives") or []},
    )
    return "" if all(checks) else "semantic_source_catalog_binding_mismatch"


def inspect_kernel_english(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = validate_kernel_english_config(config)
    disposition = validate_training_disposition(cfg)
    if disposition.get("full_kerc_training_enabled") is not True:
        return kernel_english_base_report(config_path, cfg, "GREEN", [])
    manifest_path = resolve(cfg["stage_root"]) / "manifest.json"
    if not manifest_path.is_file():
        return kernel_english_base_report(
            config_path, cfg, "PLANNED", ["kernel_english_stage_not_materialized"]
        )
    payload = read_json(manifest_path)
    gaps = validate_kernel_english_manifest(payload, cfg)
    return {
        **payload,
        "created_utc": now(),
        "mode": "inspection",
        "trigger_state": "RED" if gaps else "GREEN",
        "hard_gaps": gaps,
    }


def materialize_kernel_english(
    config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    cfg = validate_kernel_english_config(config)
    disposition = validate_training_disposition(cfg)
    if disposition.get("full_kerc_training_enabled") is not True:
        report = kernel_english_base_report(config_path, cfg, "GREEN", [])
        stage_root = resolve(cfg["stage_root"])
        stage_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(stage_root / "manifest.json", report)
        return report
    started = time.perf_counter()
    stage_root = resolve(cfg["stage_root"])
    stage_root.mkdir(parents=True, exist_ok=True)
    records_path = resolve(cfg["records_jsonl"])
    ledger_path = resolve(cfg["verification_ledger_jsonl"])
    source_catalog_path = resolve(cfg["semantic_source_catalog_json"])
    missing = []
    if not records_path.is_file():
        missing.append("kernel_english_records_missing")
    if not ledger_path.is_file():
        missing.append("kernel_english_verification_ledger_missing")
    if not source_catalog_path.is_file():
        missing.append("kernel_english_semantic_source_catalog_missing")
    if missing:
        report = kernel_english_base_report(
            config_path,
            cfg,
            "RED",
            missing,
        )
        write_json_atomic(stage_root / "manifest.json", report)
        return report

    ledger, ledger_gaps = load_kernel_verification_ledger(ledger_path)
    source_catalog, source_catalog_gaps = load_kerc_semantic_source_catalog(
        source_catalog_path, cfg
    )
    metadata = read_json(resolve(config["stage_dir"]) / "stage_metadata_v1.json")
    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    selectors = {
        split: BoundedRows(int(count))
        for split, count in cfg["records_by_split"].items()
    }
    rejection_counts: Counter[str] = Counter()
    candidate_count: Counter[str] = Counter()
    for line_number, raw in enumerate(records_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = validate_training_record(json.loads(raw))
        except Exception as exc:
            code = str(getattr(exc, "code", "KERC_RECORD_INVALID"))
            rejection_counts[code] += 1
            continue
        split = str(record["split"])
        candidate_count[split] += 1
        receipt = record["verification_receipt"]
        ledger_receipt = ledger.get(str(receipt["receipt_id"]))
        if ledger_receipt is None:
            rejection_counts["verification_receipt_absent_from_ledger"] += 1
            continue
        if ledger_receipt != receipt:
            rejection_counts["verification_receipt_ledger_mismatch"] += 1
            continue
        if str(record["provenance"]["license_spdx"]).lower() not in {
            str(value).lower() for value in cfg["allowed_licenses"]
        }:
            rejection_counts["license_not_allowed"] += 1
            continue
        source_gap = validate_kerc_record_source(record, source_catalog)
        if source_gap:
            rejection_counts[source_gap] += 1
            continue
        selectors[split].add(str(record["record_sha256"]).split(":", 1)[-1], record)

    selected = {split: selector.rows() for split, selector in selectors.items()}
    overlaps = kernel_english_split_overlap(selected)
    gaps = [*ledger_gaps, *source_catalog_gaps, *overlaps["hard_gaps"]]
    semantic_program = validate_kerc_semantic_program(cfg)
    evidence_counts_by_split: dict[str, Counter[str]] = {
        split: Counter(
            str((record.get("semantic_supervision") or {}).get("evidence_tier") or "")
            for record in records
        )
        for split, records in selected.items()
    }
    decision_grade_tiers = {
        tier
        for tier, contract in SEMANTIC_EVIDENCE_TIERS.items()
        if contract["claim_authority"] == "decision_grade_reference"
    }
    decision_grade_counts_by_split_and_objective = {
        split: {
            objective: sum(
                1
                for record in records
                if str((record.get("semantic_supervision") or {}).get("evidence_tier") or "")
                in decision_grade_tiers
                and (record.get("semantic_supervision") or {})
                .get("objective_authority", {})
                .get(objective)
                is True
            )
            for objective in TRAINING_OBJECTIVES
        }
        for split, records in selected.items()
    }
    for split, objective_floors in semantic_program[
        "minimum_decision_grade_records_by_split_and_objective"
    ].items():
        for objective, floor in objective_floors.items():
            observed = decision_grade_counts_by_split_and_objective.get(split, {}).get(
                objective, 0
            )
            if observed < int(floor):
                gaps.append(
                    f"insufficient_decision_grade_kernel_records:{split}:{objective}:"
                    f"{observed}:{int(floor)}"
                )
    train_records = selected.get("private_train") or []
    train_count = len(train_records)
    train_counts = evidence_counts_by_split.get("private_train") or Counter()
    for tier, cap in semantic_program["maximum_train_record_share_by_tier"].items():
        share = train_counts[tier] / max(1, train_count)
        if share > float(cap) + 1e-12:
            gaps.append(f"kernel_semantic_record_share_exceeded:{tier}:{share:.8f}:{cap}")
    train_weight_by_tier: Counter[str] = Counter()
    for record in train_records:
        semantic = record["semantic_supervision"]
        train_weight_by_tier[str(semantic["evidence_tier"])] += float(
            semantic["optimizer_sampling_weight"]
        )
    train_weight_total = sum(train_weight_by_tier.values())
    for tier, cap in semantic_program["maximum_train_optimizer_probability_by_tier"].items():
        probability = train_weight_by_tier[tier] / max(1e-12, train_weight_total)
        if probability > float(cap) + 1e-12:
            gaps.append(
                f"kernel_semantic_optimizer_probability_exceeded:{tier}:"
                f"{probability:.8f}:{cap}"
            )
    artifacts: dict[str, Any] = {}
    objective_counts: Counter[str] = Counter()
    objective_counts_by_split: dict[str, Counter[str]] = {
        split: Counter() for split in selected
    }
    encoded_length_stats: dict[str, Any] = {}
    all_source_hashes: set[str] = set()
    raw_source_bytes = 0
    verifier_corruption_count = 0
    compiled_views = {
        split: [
            view
            for record in records
            for view in compile_training_views(record)
        ]
        for split, records in selected.items()
    }
    if not compiled_views.get("private_train"):
        raise ValueError(
            "KERC stage has no admitted private-train views: "
            + json.dumps(dict(rejection_counts), sort_keys=True)
        )
    code_vocabulary = build_kerc_code_vocabulary(
        compiled_views["private_train"], cfg["code_vocabulary"]
    )
    code_vocabulary_path = stage_root / "code_vocabulary_v1.json"
    write_json_atomic(code_vocabulary_path, code_vocabulary)
    for split, records in selected.items():
        wanted = int(cfg["records_by_split"][split])
        if len(records) != wanted:
            gaps.append(f"insufficient_kernel_records:{split}:{len(records)}:{wanted}")
        views: list[dict[str, Any]] = []
        source_lengths: list[int] = []
        target_lengths: list[int] = []
        sequence_lengths: list[int] = []
        for record in records:
            all_source_hashes.add(str(record["raw_source_sha256"]))
            raw_source_bytes += len(str(record["source_text"]).encode("utf-8"))
            for view in compile_training_views(record):
                source_body_ids, source_receipt = encode_tokens(
                    kerc_surface_tokens(view["prompt"]), source_vocab, stream="source"
                )
                trusted_prefix = list(view.get("trusted_source_prefix_tokens") or [])
                if len(trusted_prefix) != 1 or trusted_prefix[0] not in source_vocab:
                    gaps.append(f"kernel_view_trusted_prefix_invalid:{view['row_id']}")
                    continue
                source_ids = [int(source_vocab[trusted_prefix[0]]), *source_body_ids]
                target_ids, target_receipt = encode_kerc_view_target(
                    view, target_vocab=target_vocab, code_vocabulary=code_vocabulary
                )
                if int(source_receipt.get("unknown_token_count") or 0) or int(
                    target_receipt.get("unknown_token_count") or 0
                ):
                    gaps.append(f"kernel_view_unrepresentable:{view['row_id']}")
                    continue
                verifier_negative = view.get("kerc_verifier_negative") or {}
                negative_target = str(verifier_negative.get("target") or "")
                negative_ids, negative_receipt = encode_kerc_view_target(
                    {**view, "target": negative_target},
                    target_vocab=target_vocab,
                    code_vocabulary=code_vocabulary,
                )
                if (
                    not negative_target
                    or verifier_negative.get("generator_loss_enabled") is not False
                    or int(negative_receipt.get("unknown_token_count") or 0)
                ):
                    gaps.append(
                        f"kernel_view_verifier_corruption_invalid:{view['row_id']}"
                    )
                    continue
                sequence_tokens = len(source_ids) + len(target_ids) + 4
                negative_sequence_tokens = len(source_ids) + len(negative_ids) + 4
                if sequence_tokens > int(cfg["maximum_sequence_tokens"]):
                    gaps.append(
                        f"kernel_view_requires_truncation:{view['row_id']}:{sequence_tokens}"
                    )
                    continue
                if negative_sequence_tokens > int(cfg["maximum_sequence_tokens"]):
                    gaps.append(
                        "kernel_view_verifier_corruption_requires_truncation:"
                        f"{view['row_id']}:{negative_sequence_tokens}"
                    )
                    continue
                source_lengths.append(len(source_ids))
                target_lengths.append(len(target_ids))
                target_lengths.append(len(negative_ids))
                sequence_lengths.extend((sequence_tokens, negative_sequence_tokens))
                objective_counts[str(view["objective"])] += 1
                objective_counts_by_split[split][str(view["objective"])] += 1
                verifier_corruption_count += 1
                views.append(view)
        path = stage_root / f"{split}.jsonl"
        write_jsonl_atomic(path, views)
        artifacts[f"english:{split}"] = {
            "path": relative(path),
            "sha256": sha256_file(path),
            "row_count": len(views),
            "unique_record_count": len(records),
            "bytes": path.stat().st_size,
        }
        encoded_length_stats[split] = {
            "maximum_source_tokens": max(source_lengths or [0]),
            "maximum_target_tokens": max(target_lengths or [0]),
            "maximum_sequence_tokens": max(sequence_lengths or [0]),
        }

    report = {
        "policy": cfg["policy"],
        "created_utc": now(),
        "mode": "materialized",
        "trigger_state": "RED" if gaps else "GREEN",
        "config": relative(config_path),
        "contract_sha256": kernel_english_stage_contract_sha256(cfg),
        "learned_pipeline_contract": kernel_training_contract(),
        "required_records_by_split": dict(cfg["records_by_split"]),
        "verification_ledger_required": True,
        "source": {
            "path": relative(records_path),
            "sha256": sha256_file(records_path),
            "license_policy": "row_level_explicit_allowlist",
        },
        "verification_ledger": {
            "path": relative(ledger_path),
            "sha256": sha256_file(ledger_path),
            "receipt_count": len(ledger),
            "producer_separate_from_training_rows": True,
        },
        "semantic_source_catalog": {
            "path": relative(source_catalog_path),
            "sha256": sha256_file(source_catalog_path),
            "policy": KERC_SOURCE_CATALOG_POLICY,
            "source_count": len(source_catalog),
        },
        "semantic_supervision": {
            "policy": KERC_SEMANTIC_PROGRAM_POLICY,
            "evidence_record_counts_by_split": {
                split: dict(counts) for split, counts in evidence_counts_by_split.items()
            },
            "decision_grade_record_counts_by_split_and_objective": (
                decision_grade_counts_by_split_and_objective
            ),
            "minimum_decision_grade_records_by_split_and_objective": dict(
                semantic_program["minimum_decision_grade_records_by_split_and_objective"]
            ),
            "train_weight_by_tier": dict(train_weight_by_tier),
            "train_optimizer_probability_by_tier": {
                tier: round(weight / max(1e-12, train_weight_total), 10)
                for tier, weight in train_weight_by_tier.items()
            },
            "silver_supports_decision_grade_claims": False,
            "teacher_residual_supports_decision_grade_claims": False,
        },
        "code_vocabulary": {
            "path": relative(code_vocabulary_path),
            "sha256": sha256_file(code_vocabulary_path),
            "policy": code_vocabulary["policy"],
            "contract_sha256": code_vocabulary["contract_sha256"],
            "fit_split": code_vocabulary["fit_split"],
            "kernel_vocab_count": len(code_vocabulary["kernel_vocab"]),
            "pointer_vocab_count": len(code_vocabulary["pointer_vocab"]),
            "kernel_max_vocab": code_vocabulary["kernel_max_vocab"],
            "pointer_max_vocab": code_vocabulary["pointer_max_vocab"],
        },
        "artifacts": artifacts,
        "candidate_record_count_by_split": dict(candidate_count),
        "selected_record_count_by_split": {
            split: len(records) for split, records in selected.items()
        },
        "compiled_view_count_by_objective": dict(objective_counts),
        "compiled_view_count_by_split_and_objective": {
            split: dict(counts) for split, counts in objective_counts_by_split.items()
        },
        "unique_raw_source_count": len(all_source_hashes),
        "unique_raw_source_bytes": raw_source_bytes,
        "derived_view_unique_data_credit": 0,
        "derived_view_optimizer_exposure_count": sum(objective_counts.values()),
        "verifier_corruption_count": verifier_corruption_count,
        "verifier_corruptions_receive_generator_loss": False,
        "split_overlap_audit": overlaps,
        "encoded_length_stats": encoded_length_stats,
        "rejection_counts": dict(rejection_counts),
        "failure_behavior": "reject_without_template_literal_tool_or_router_fallback",
        "score_semantics": "KERC learned-objective data readiness; not learned capability",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "hard_gaps": sorted(set(gaps)),
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "deterministic_renderer_credit": 0,
        "candidate_generation_credit": 0,
    }
    write_json_atomic(stage_root / "manifest.json", report)
    return report


def kernel_english_split_overlap(
    selected: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    groups: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    for split, records in selected.items():
        groups[split] = {str(row["provenance"]["source_group"]) for row in records}
        sources[split] = {str(row["raw_source_sha256"]) for row in records}
    group_overlap = 0
    source_overlap = 0
    for left_index, left in enumerate(selected):
        for right in tuple(selected)[left_index + 1 :]:
            group_overlap += len(groups[left] & groups[right])
            source_overlap += len(sources[left] & sources[right])
    gaps = []
    if group_overlap:
        gaps.append(f"kernel_source_group_cross_split_overlap:{group_overlap}")
    if source_overlap:
        gaps.append(f"kernel_raw_source_cross_split_overlap:{source_overlap}")
    return {
        "source_group_overlap_count": group_overlap,
        "raw_source_overlap_count": source_overlap,
        "content_bound_disjoint": not gaps,
        "hard_gaps": gaps,
    }


def load_kernel_verification_ledger(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    receipts: dict[str, dict[str, Any]] = {}
    gaps: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            gaps.append(f"kernel_verification_ledger_json_invalid:{line_number}")
            continue
        receipt_id = str(row.get("receipt_id") or "") if isinstance(row, dict) else ""
        if not receipt_id:
            gaps.append(f"kernel_verification_ledger_receipt_id_missing:{line_number}")
            continue
        if receipt_id in receipts:
            gaps.append(f"kernel_verification_ledger_receipt_duplicate:{receipt_id}")
            continue
        if row.get("policy") != TRAINING_VERIFICATION_POLICY:
            gaps.append(f"kernel_verification_ledger_policy_invalid:{receipt_id}")
            continue
        if row.get("accepted") is not True:
            gaps.append(f"kernel_verification_ledger_unaccepted:{receipt_id}")
            continue
        receipts[receipt_id] = row
    return receipts, sorted(set(gaps))


def validate_kernel_english_manifest(
    payload: dict[str, Any], cfg: dict[str, Any]
) -> list[str]:
    gaps: list[str] = []
    if payload.get("policy") != cfg["policy"]:
        gaps.append("kernel_stage_policy_mismatch")
    if payload.get("contract_sha256") != kernel_english_stage_contract_sha256(cfg):
        gaps.append("kernel_stage_contract_identity_mismatch")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    expected_view_count = 0
    for split, record_count in cfg["records_by_split"].items():
        key = f"english:{split}"
        artifact = artifacts.get(key) if isinstance(artifacts.get(key), dict) else {}
        path = resolve(str(artifact.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(artifact.get("sha256") or ""):
            gaps.append(f"kernel_stage_artifact_identity_mismatch:{key}")
        row_count = int(artifact.get("row_count") or 0)
        if not int(record_count) <= row_count <= int(record_count) * len(TRAINING_OBJECTIVES):
            gaps.append(f"kernel_stage_view_count_mismatch:{key}")
        expected_view_count += row_count
        if int(artifact.get("unique_record_count") or 0) != int(record_count):
            gaps.append(f"kernel_stage_record_count_mismatch:{key}")
    overlap = payload.get("split_overlap_audit") or {}
    if not bool(overlap.get("content_bound_disjoint")):
        gaps.append("kernel_stage_split_overlap")
    if int(payload.get("derived_view_unique_data_credit") or 0):
        gaps.append("kernel_stage_derived_view_unique_credit_nonzero")
    if int(payload.get("verifier_corruption_count") or 0) != expected_view_count:
        gaps.append("kernel_stage_verifier_corruption_count_mismatch")
    if payload.get("verifier_corruptions_receive_generator_loss") is not False:
        gaps.append("kernel_stage_verifier_corruption_generator_credit")
    ledger = payload.get("verification_ledger") or {}
    ledger_path = resolve(str(ledger.get("path") or ""))
    if (
        not ledger_path.is_file()
        or sha256_file(ledger_path) != str(ledger.get("sha256") or "")
    ):
        gaps.append("kernel_stage_verification_ledger_identity_mismatch")
    if ledger.get("producer_separate_from_training_rows") is not True:
        gaps.append("kernel_stage_verification_ledger_not_independent")
    source_catalog = payload.get("semantic_source_catalog") or {}
    source_catalog_path = resolve(str(source_catalog.get("path") or ""))
    if (
        not source_catalog_path.is_file()
        or sha256_file(source_catalog_path) != str(source_catalog.get("sha256") or "")
        or source_catalog.get("policy") != KERC_SOURCE_CATALOG_POLICY
    ):
        gaps.append("kernel_stage_semantic_source_catalog_identity_mismatch")
    semantic = payload.get("semantic_supervision") or {}
    program = validate_kerc_semantic_program(cfg)
    if semantic.get("policy") != KERC_SEMANTIC_PROGRAM_POLICY:
        gaps.append("kernel_stage_semantic_supervision_policy_mismatch")
    if semantic.get("minimum_decision_grade_records_by_split_and_objective") != program.get(
        "minimum_decision_grade_records_by_split_and_objective"
    ):
        gaps.append("kernel_stage_decision_grade_floor_mismatch")
    decision_counts = (
        semantic.get("decision_grade_record_counts_by_split_and_objective") or {}
    )
    for split, objective_floors in program[
        "minimum_decision_grade_records_by_split_and_objective"
    ].items():
        for objective, floor in objective_floors.items():
            if int((decision_counts.get(split) or {}).get(objective) or 0) < int(floor):
                gaps.append(
                    f"kernel_stage_decision_grade_floor_not_met:{split}:{objective}"
                )
    if semantic.get("silver_supports_decision_grade_claims") is not False:
        gaps.append("kernel_stage_silver_claim_authority_invalid")
    if semantic.get("teacher_residual_supports_decision_grade_claims") is not False:
        gaps.append("kernel_stage_teacher_claim_authority_invalid")
    code = payload.get("code_vocabulary") or {}
    code_path = resolve(str(code.get("path") or ""))
    code_payload = read_json(code_path) if code_path.is_file() else {}
    if (
        not code_path.is_file()
        or sha256_file(code_path) != str(code.get("sha256") or "")
        or code_payload.get("policy")
        != "project_theseus_kerc_dual_code_vocabulary_v1"
        or code_payload.get("fit_split") != "private_train"
        or int(code_payload.get("dev_eval_vocabulary_fit_count") or 0) != 0
        or int(code_payload.get("verifier_corruption_vocabulary_fit_count") or 0) != 0
        or code_payload.get("contract_sha256") != code.get("contract_sha256")
    ):
        gaps.append("kernel_stage_code_vocabulary_identity_mismatch")
    else:
        unsigned = {
            key: value for key, value in code_payload.items() if key != "contract_sha256"
        }
        observed_contract = "sha256:" + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if observed_contract != code_payload["contract_sha256"]:
            gaps.append("kernel_stage_code_vocabulary_contract_mismatch")
        if (
            len(code_payload.get("kernel_vocab") or {})
            != int(code.get("kernel_vocab_count") or 0)
            or len(code_payload.get("pointer_vocab") or {})
            != int(code.get("pointer_vocab_count") or 0)
        ):
            gaps.append("kernel_stage_code_vocabulary_count_mismatch")
    for key in (
        "public_training_rows_written",
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
        "template_credit",
        "deterministic_renderer_credit",
        "candidate_generation_credit",
    ):
        if int(payload.get(key) or 0):
            gaps.append(f"kernel_stage_nonzero_boundary:{key}")
    return sorted(set(gaps))


def kernel_english_stage_contract_sha256(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def kernel_english_base_report(
    config_path: Path,
    cfg: dict[str, Any],
    state: str,
    gaps: list[str],
) -> dict[str, Any]:
    disposition = validate_training_disposition(cfg)
    enabled = disposition.get("full_kerc_training_enabled") is True
    return {
        "policy": cfg["policy"],
        "created_utc": now(),
        "mode": "inspection" if enabled else "retired_from_first_long_run",
        "trigger_state": state,
        "config": relative(config_path),
        "contract_sha256": kernel_english_stage_contract_sha256(cfg),
        "learned_pipeline_contract": kernel_training_contract() if enabled else {},
        "architecture_disposition": disposition,
        "full_kerc_training_enabled": enabled,
        "retained_mechanisms": list(disposition.get("retained_mechanisms") or []),
        "required_records_by_split": dict(cfg["records_by_split"]),
        "verification_ledger_required": enabled,
        "artifacts": {},
        "selected_record_count_by_split": {
            split: 0 for split in cfg["records_by_split"]
        },
        "compiled_view_count_by_objective": {},
        "unique_raw_source_count": 0,
        "derived_view_unique_data_credit": 0,
        "split_overlap_audit": {
            "source_group_overlap_count": 0,
            "raw_source_overlap_count": 0,
            "content_bound_disjoint": True,
            "hard_gaps": [],
        },
        "hard_gaps": gaps,
        "score_semantics": (
            "KERC learned-objective data readiness; not learned capability"
            if enabled
            else "bounded pre-training architecture disposition; full KERC receives zero optimizer exposure"
        ),
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "deterministic_renderer_credit": 0,
        "candidate_generation_credit": 0,
    }


def inspect(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = validate_config(config)
    manifest_path = resolve(cfg["stage_root"]) / "manifest.json"
    if not manifest_path.is_file():
        return base_report(config_path, cfg, "PLANNED", ["stage_not_materialized"])
    payload = read_json(manifest_path)
    gaps = validate_manifest(payload, cfg, config)
    return {
        **payload,
        "created_utc": now(),
        "mode": "inspection",
        "trigger_state": "RED" if gaps else "GREEN",
        "hard_gaps": gaps,
    }


def materialize(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = validate_config(config)
    started = time.perf_counter()
    stage_root = resolve(cfg["stage_root"])
    stage_root.mkdir(parents=True, exist_ok=True)
    dependencies = source_conditioning_dependencies(config)
    metadata = read_json(resolve(config["stage_dir"]) / "stage_metadata_v1.json")
    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    supervision_targets = supervision_target_hashes(config)
    selectors = {
        arm: BoundedRows(int(count))
        for arm, count in cfg["rows_by_arm"].items()
        if int(count) > 0
    }
    language_to_arm = {
        language: arm
        for arm, languages in (cfg.get("arm_languages") or {}).items()
        for language in languages
    }
    source_path = resolve(cfg["source_jsonl"])
    rejections: Counter[str] = Counter()
    candidate_count: Counter[str] = Counter()
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            arm = language_to_arm.get(str(source.get("language") or "").lower())
            if arm not in selectors:
                continue
            reason = source_rejection(source, cfg)
            if reason:
                rejections[reason] += 1
                continue
            for row in denoising_rows(source, arm, cfg, source_vocab, target_vocab):
                candidate_count[arm] += 1
                if row["target_sha256"] in supervision_targets:
                    rejections["supervision_target_overlap"] += 1
                    continue
                selectors[arm].add(row["selection_sha256"], row)

    artifacts: dict[str, Any] = {}
    copy_coverage: dict[str, Any] = {}
    gaps: list[str] = []
    for arm, selector in selectors.items():
        rows = selector.rows()
        wanted = int(cfg["rows_by_arm"][arm])
        if len(rows) != wanted:
            gaps.append(f"insufficient_rows:{arm}:{len(rows)}:{wanted}")
        path = stage_root / f"{arm}.jsonl"
        write_jsonl_atomic(path, rows)
        artifacts[arm] = {
            "path": relative(path),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "bytes": path.stat().st_size,
        }
        fractions = [float(row["target_token_copy_fraction"]) for row in rows]
        copy_coverage[arm] = {
            "mean_target_token_copy_fraction": round(
                sum(fractions) / max(1, len(fractions)), 8
            ),
            "minimum_target_token_copy_fraction": round(min(fractions or [0.0]), 8),
        }
    report = {
        "policy": cfg["policy"],
        "created_utc": now(),
        "mode": "materialized",
        "trigger_state": "RED" if gaps else "GREEN",
        "config": relative(config_path),
        "contract_sha256": contract_sha256(cfg),
        "dependencies": dependencies,
        "source": {
            "path": relative(source_path),
            "sha256": sha256_file(source_path),
            "license_policy": "row_level_permissive_allowlist",
        },
        "artifacts": artifacts,
        "candidate_count_by_arm": dict(candidate_count),
        "copy_coverage_by_arm": copy_coverage,
        "rejection_counts": dict(rejections),
        "supervision_target_overlap_count": int(rejections["supervision_target_overlap"]),
        "corruption": {
            "mode": "deterministic_span_deletion_reconstruction",
            "deletion_fraction": float(cfg["deletion_fraction"]),
            "maximum_spans": int(cfg["maximum_deletion_spans"]),
            "seed": int(cfg["seed"]),
        },
        "generator_visible_fields": ["prompt"],
        "evaluator_only_fields": ["target", "target_sha256", "source_identity"],
        "score_semantics": "licensed source-conditioned objective readiness; not edit capability",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "hard_gaps": gaps,
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    write_json_atomic(stage_root / "manifest.json", report)
    return report


def denoising_rows(
    source: dict[str, Any],
    arm: str,
    cfg: dict[str, Any],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
) -> list[dict[str, Any]]:
    text = str(source.get("text") or "")
    logical = exact_text_tokens(text)
    minimum = int(cfg["minimum_target_logical_tokens"])
    maximum = int(cfg["maximum_target_logical_tokens"])
    if len(logical) < minimum:
        return []
    source_identity = str(source.get("text_sha256") or hashlib.sha256(text.encode()).hexdigest())
    starts = list(range(0, len(logical) - minimum + 1, maximum))
    ranked_starts = sorted(
        starts,
        key=lambda start: hashlib.sha256(f"{source_identity}:{start}:{cfg['seed']}".encode()).hexdigest(),
    )[: int(cfg["maximum_windows_per_document"])]
    rows = []
    for start in ranked_starts:
        target_tokens = logical[start : start + maximum]
        if len(target_tokens) < minimum:
            continue
        corruption_identity = hashlib.sha256(
            f"{source_identity}:{start}:{cfg['seed']}".encode()
        ).hexdigest()
        damaged_tokens = delete_spans(target_tokens, cfg, corruption_identity)
        target = "".join(target_tokens)
        damaged = "".join(damaged_tokens)
        if not target.strip() or damaged == target:
            continue
        language = str(source.get("language") or arm)
        prompt = (
            f"Reconstruct the complete original {language} excerpt from this damaged excerpt. "
            "Return only the original excerpt.\n\n"
            f"Damaged excerpt:\n{damaged}"
        )
        source_ids, source_receipt = encode_tokens(
            exact_text_tokens(prompt), source_vocab, stream="source"
        )
        target_ids, target_receipt = encode_tokens(
            exact_text_tokens(target), target_vocab, stream="target"
        )
        if int(source_receipt.get("unknown_token_count") or 0) or int(
            target_receipt.get("unknown_token_count") or 0
        ):
            continue
        if len(source_ids) > int(cfg["maximum_source_encoded_tokens"]) or len(
            target_ids
        ) > int(cfg["maximum_target_encoded_tokens"]):
            continue
        source_token_set = set(exact_text_tokens(prompt))
        copy_fraction = sum(token in source_token_set for token in exact_text_tokens(target)) / max(
            1, len(exact_text_tokens(target))
        )
        digest = hashlib.sha256(
            f"{arm}:{source_identity}:{start}:{corruption_identity}".encode()
        ).hexdigest()
        rows.append(
            {
                "row_id": f"moecot-denoise-{digest[:20]}",
                "split": "private_train",
                "arm_id": arm,
                "objective": "source_conditioned_span_deletion_reconstruction_v1",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "target": target,
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                "target_token_copy_fraction": round(copy_fraction, 8),
                "selection_sha256": digest,
                "source_identity": {
                    "repo": source.get("repo"),
                    "path": source.get("path"),
                    "text_sha256": source_identity,
                    "window_start": start,
                    "license_spdx": source.get("license_spdx"),
                },
                "public_benchmark": False,
                "public_tests_included": False,
                "public_benchmark_solutions_included": False,
                "external_inference": False,
            }
        )
    return rows


def delete_spans(tokens: list[str], cfg: dict[str, Any], identity: str) -> list[str]:
    rng = random.Random(int(identity[:16], 16))
    delete_count = max(1, round(len(tokens) * float(cfg["deletion_fraction"])))
    spans = min(int(cfg["maximum_deletion_spans"]), delete_count)
    removed: set[int] = set()
    remaining = delete_count
    for span_index in range(spans):
        width = max(1, remaining // (spans - span_index))
        start = rng.randrange(max(1, len(tokens) - width + 1))
        removed.update(range(start, min(len(tokens), start + width)))
        remaining = max(0, delete_count - len(removed))
    while len(removed) < delete_count:
        removed.add(rng.randrange(len(tokens)))
    return [token for index, token in enumerate(tokens) if index not in removed]


def source_rejection(source: dict[str, Any], cfg: dict[str, Any]) -> str:
    if source.get("public_benchmark") is not False:
        return "public_benchmark_state_not_false"
    if source.get("public_tests_included") is not False:
        return "public_tests_present"
    if source.get("public_benchmark_solutions_included") is not False:
        return "public_solutions_present"
    if str(source.get("license_spdx") or "").lower() not in {
        str(value).lower() for value in cfg["allowed_licenses"]
    }:
        return "license_not_allowed"
    if not str(source.get("text") or "").strip():
        return "empty_text"
    return ""


def supervision_target_hashes(config: dict[str, Any]) -> set[str]:
    root = resolve(config["supervision"]["stage_root"])
    hashes: set[str] = set()
    for path in sorted(root.glob("private_*/*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                hashes.add(str(row.get("target_sha256") or ""))
    return hashes


def source_conditioning_dependencies(config: dict[str, Any]) -> dict[str, Any]:
    """Bind every mutable input that changes source-conditioned row selection."""

    cfg = validate_config(config)
    source_path = resolve(cfg["source_jsonl"])
    metadata_path = resolve(config["stage_dir"]) / "stage_metadata_v1.json"
    supervision_root = resolve(config["supervision"]["stage_root"])
    supervision_paths = [
        path
        for path in [
            supervision_root / "manifest.json",
            *sorted(supervision_root.glob("private_*/*.jsonl")),
        ]
        if path.is_file()
    ]
    supervision_files = [
        {
            "path": relative(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in supervision_paths
    ]
    supervision_digest = hashlib.sha256(
        json.dumps(supervision_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "source_jsonl": {
            "path": relative(source_path),
            "sha256": sha256_file(source_path),
            "bytes": source_path.stat().st_size,
        },
        "stage_metadata": {
            "path": relative(metadata_path),
            "sha256": sha256_file(metadata_path),
            "bytes": metadata_path.stat().st_size,
        },
        "supervision_stage": {
            "root": relative(supervision_root),
            "file_count": len(supervision_files),
            "files": supervision_files,
            "sha256": supervision_digest,
        },
    }


def contract_sha256(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_manifest(
    payload: dict[str, Any], cfg: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    gaps = []
    if payload.get("policy") != cfg["policy"]:
        gaps.append("policy_mismatch")
    if payload.get("contract_sha256") != contract_sha256(cfg):
        gaps.append("contract_identity_mismatch")
    recorded_dependencies = payload.get("dependencies") or {}
    try:
        current_dependencies = source_conditioning_dependencies(config)
    except (FileNotFoundError, KeyError, OSError) as exc:
        gaps.append(f"dependency_identity_unavailable:{type(exc).__name__}")
        current_dependencies = {}
    for dependency in ("source_jsonl", "stage_metadata", "supervision_stage"):
        if recorded_dependencies.get(dependency) != current_dependencies.get(dependency):
            gaps.append(f"dependency_identity_mismatch:{dependency}")
    for arm, wanted in cfg["rows_by_arm"].items():
        if int(wanted) <= 0:
            continue
        artifact = (payload.get("artifacts") or {}).get(arm) or {}
        path = resolve(str(artifact.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(artifact.get("sha256") or ""):
            gaps.append(f"artifact_identity_mismatch:{arm}")
        if int(artifact.get("row_count") or 0) != int(wanted):
            gaps.append(f"row_count_mismatch:{arm}")
    for key in ("public_training_rows_written", "public_benchmark_payload_count", "external_inference_calls", "fallback_return_count"):
        if int(payload.get(key) or 0):
            gaps.append(f"nonzero_boundary:{key}")
    return gaps


def base_report(
    config_path: Path, cfg: dict[str, Any], state: str, gaps: list[str]
) -> dict[str, Any]:
    return {
        "policy": cfg["policy"],
        "created_utc": now(),
        "mode": "inspection",
        "trigger_state": state,
        "config": relative(config_path),
        "contract_sha256": contract_sha256(cfg),
        "hard_gaps": gaps,
        "score_semantics": "licensed source-conditioned objective readiness; not capability",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
