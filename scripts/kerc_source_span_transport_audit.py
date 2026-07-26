#!/usr/bin/env python3
"""Audit KERC v4 source-span transport over frozen admitted compiler rows.

This is a transport and construct-validity audit only. It does not generate a
candidate, evaluate capability, or grant deterministic materialization credit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kernel_english_protocol import (
    LEARNED_COMPILER_SEMANTIC_POINTER_TRANSPORT_VERSION,
    LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_POLICY,
    LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION,
    canonical_json,
    compact_learned_compiler_transport,
    materialize_learned_compiler_transport,
    stable_hash,
)
from moecot_source_conditioned_pretraining import (
    encode_kerc_global_target,
    kerc_code_tokens,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = (
    ROOT / "data" / "training_data" / "moecot_kernel_english_v1" / "private_train.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "reports" / "rdc_kerc_k5_source_span_transport_v4_population_audit.json"
)
DEFAULT_CODE_VOCABULARY = (
    ROOT
    / "data"
    / "training_data"
    / "moecot_kernel_english_v1"
    / "code_vocabulary_v1.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compiler_source(prompt: str, *, identity: str) -> str:
    try:
        decoded = json.loads(prompt)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{identity}: compiler prompt is not JSON") from exc
    source = str(decoded.get("source_surface") or "")
    if not source or decoded.get("source_character_length") != len(source):
        raise ValueError(f"{identity}: compiler source contract is invalid")
    return source


def _target_variants(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    identity = str(row.get("row_id") or "")
    source = _compiler_source(str(row.get("prompt") or ""), identity=identity)
    variants = [("positive", str(row.get("target") or ""), source)]
    negative = row.get("kerc_verifier_negative") or {}
    if negative.get("target"):
        variants.append(("verifier_negative", str(negative["target"]), source))
    for index, counterfactual in enumerate(row.get("kerc_context_counterfactuals") or []):
        counter_source = _compiler_source(
            str(counterfactual.get("prompt") or ""),
            identity=f"{identity}:counterfactual:{index}",
        )
        variants.append(
            (
                f"context_counterfactual:{index}",
                str(counterfactual.get("target") or ""),
                counter_source,
            )
        )
    return variants


def audit_artifact(
    artifact: Path,
    *,
    code_vocabulary_path: Path = DEFAULT_CODE_VOCABULARY,
    maximum_compiler_rows: int = 0,
) -> dict[str, Any]:
    code_vocabulary = json.loads(code_vocabulary_path.read_text())
    counters = {
        "artifact_rows": 0,
        "compiler_rows": 0,
        "target_variants": 0,
        "positive_targets": 0,
        "verifier_negative_targets": 0,
        "context_counterfactual_targets": 0,
        "legacy_byte_literal_tokens": 0,
        "source_span_pointer_tokens": 0,
        "retained_non_pointer_byte_literal_tokens": 0,
        "exact_v3_semantic_equivalence_targets": 0,
        "roundtrip_failures": 0,
        "v3_logical_code_tokens": 0,
        "v4_logical_code_tokens": 0,
        "v3_encoded_target_tokens": 0,
        "v4_encoded_target_tokens": 0,
        "v3_fallback_target_tokens": 0,
        "v4_fallback_target_tokens": 0,
        "v3_wire_bytes": 0,
        "v4_wire_bytes": 0,
    }
    row_ids: list[str] = []
    failures: list[dict[str, Any]] = []
    with artifact.open(encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            counters["artifact_rows"] += 1
            row = json.loads(line)
            if row.get("objective") != "surface_to_kernel_program_v1":
                continue
            if maximum_compiler_rows and counters["compiler_rows"] >= maximum_compiler_rows:
                break
            if row.get("split") != "private_train" or row.get("public_benchmark") is not False:
                raise ValueError(f"row {source_index}: compiler admission boundary is invalid")
            counters["compiler_rows"] += 1
            row_id = str(row.get("row_id") or "")
            row_ids.append(row_id)
            for kind, target_text, source in _target_variants(row):
                counters["target_variants"] += 1
                if kind == "positive":
                    counters["positive_targets"] += 1
                elif kind == "verifier_negative":
                    counters["verifier_negative_targets"] += 1
                else:
                    counters["context_counterfactual_targets"] += 1
                try:
                    legacy = json.loads(target_text)
                    legacy_tokens = list((legacy.get("program") or {}).get("tokens") or [])
                    counters["legacy_byte_literal_tokens"] += sum(
                        isinstance(token, str) and token.startswith("PBYTE:")
                        for token in legacy_tokens
                    )
                    v3 = compact_learned_compiler_transport(
                        legacy,
                        transport_version=(
                            LEARNED_COMPILER_SEMANTIC_POINTER_TRANSPORT_VERSION
                        ),
                    )
                    expected = materialize_learned_compiler_transport(v3)
                    v4 = compact_learned_compiler_transport(
                        legacy,
                        transport_version=LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION,
                        source=source,
                    )
                    counters["source_span_pointer_tokens"] += sum(
                        isinstance(token, str)
                        and token.startswith("PSOURCE_BYTES:")
                        for token in v4[3]
                    )
                    counters["retained_non_pointer_byte_literal_tokens"] += sum(
                        isinstance(token, str) and token.startswith("PBYTE:")
                        for token in v4[3]
                    )
                    observed = materialize_learned_compiler_transport(v4, source=source)
                    if observed != expected:
                        raise ValueError("v4 materialization differs from v3 semantics")
                    counters["exact_v3_semantic_equivalence_targets"] += 1
                    v3_text = canonical_json(v3)
                    v4_text = canonical_json(v4)
                    counters["v3_wire_bytes"] += len(v3_text.encode("utf-8"))
                    counters["v4_wire_bytes"] += len(v4_text.encode("utf-8"))
                    counters["v3_logical_code_tokens"] += len(kerc_code_tokens(v3_text))
                    counters["v4_logical_code_tokens"] += len(kerc_code_tokens(v4_text))
                    v3_ids, v3_encoding = encode_kerc_global_target(
                        v3_text,
                        code_vocabulary=code_vocabulary,
                        kernel_offset=0,
                        pointer_offset=100_000,
                    )
                    v4_ids, v4_encoding = encode_kerc_global_target(
                        v4_text,
                        code_vocabulary=code_vocabulary,
                        kernel_offset=0,
                        pointer_offset=100_000,
                    )
                    if (
                        v3_encoding["unknown_token_count"]
                        or v4_encoding["unknown_token_count"]
                    ):
                        raise ValueError("frozen code vocabulary cannot encode transport")
                    counters["v3_encoded_target_tokens"] += len(v3_ids)
                    counters["v4_encoded_target_tokens"] += len(v4_ids)
                    counters["v3_fallback_target_tokens"] += int(
                        v3_encoding["fallback_token_count"]
                    )
                    counters["v4_fallback_target_tokens"] += int(
                        v4_encoding["fallback_token_count"]
                    )
                except Exception as exc:
                    counters["roundtrip_failures"] += 1
                    if len(failures) < 20:
                        failures.append(
                            {
                                "source_index": source_index,
                                "row_id": row_id,
                                "target_kind": kind,
                                "fault_type": type(exc).__name__,
                                "fault": str(exc),
                            }
                        )
    pointer_denominator = counters["legacy_byte_literal_tokens"]
    target_denominator = counters["target_variants"]
    full_population = maximum_compiler_rows == 0
    green = bool(
        full_population
        and counters["compiler_rows"] > 0
        and counters["roundtrip_failures"] == 0
        and counters["exact_v3_semantic_equivalence_targets"] == target_denominator
        and counters["source_span_pointer_tokens"]
        + counters["retained_non_pointer_byte_literal_tokens"]
        == pointer_denominator
    )
    report: dict[str, Any] = {
        "policy": "project_theseus_kerc_source_span_transport_population_audit_v1",
        "transport": {
            "policy": LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_POLICY,
            "version": LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION,
            "source_authority": "generator_visible_prompt.source_surface_only",
            "pointer_contract": (
                "model_emits_typed_character_span_runtime_materializes_exact_utf8_bytes"
            ),
            "out_of_bounds_rejected": True,
            "ambiguous_unaligned_source_slices_rejected": True,
            "non_source_byte_literals_retained": True,
        },
        "artifact": {
            "path": str(artifact.relative_to(ROOT) if artifact.is_relative_to(ROOT) else artifact),
            "sha256": sha256_file(artifact),
        },
        "code_vocabulary": {
            "path": str(
                code_vocabulary_path.relative_to(ROOT)
                if code_vocabulary_path.is_relative_to(ROOT)
                else code_vocabulary_path
            ),
            "sha256": sha256_file(code_vocabulary_path),
            "contract_sha256": code_vocabulary.get("contract_sha256"),
        },
        "full_population": full_population,
        "maximum_compiler_rows": int(maximum_compiler_rows),
        "counts": counters,
        "rates": {
            "source_pointer_share_of_byte_literals": round(
                counters["source_span_pointer_tokens"] / max(1, pointer_denominator),
                8,
            ),
            "exact_v3_semantic_equivalence_share": round(
                counters["exact_v3_semantic_equivalence_targets"]
                / max(1, target_denominator),
                8,
            ),
            "code_token_reduction_share": round(
                (
                    counters["v3_encoded_target_tokens"]
                    - counters["v4_encoded_target_tokens"]
                )
                / max(1, counters["v3_encoded_target_tokens"]),
                8,
            ),
            "wire_byte_reduction_share": round(
                (counters["v3_wire_bytes"] - counters["v4_wire_bytes"])
                / max(1, counters["v3_wire_bytes"]),
                8,
            ),
        },
        "row_id_sha256": stable_hash(row_ids),
        "failures": failures,
        "qualification": (
            "GREEN_FOR_BOUNDED_OVERFIT_CONTROL_ONLY"
            if green
            else "NOT_QUALIFIED"
        ),
        "capability_claim": "NONE",
        "learned_generation_claim": "NONE",
        "candidate_generation_credit": 0,
        "deterministic_materialization_generation_credit": 0,
        "deterministic_materialization_capability_credit": 0,
        "public_evaluation_rows_consumed": 0,
        "private_evaluation_rows_consumed": 0,
        "new_training_rows_created": 0,
        "negative_evidence_scope": (
            "transport_reversibility_and_population_compatibility_only"
        ),
    }
    report["report_sha256"] = stable_hash(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--code-vocabulary",
        type=Path,
        default=DEFAULT_CODE_VOCABULARY,
    )
    parser.add_argument("--maximum-compiler-rows", type=int, default=0)
    args = parser.parse_args()
    if args.maximum_compiler_rows < 0:
        raise SystemExit("--maximum-compiler-rows cannot be negative")
    report = audit_artifact(
        args.artifact.resolve(),
        code_vocabulary_path=args.code_vocabulary.resolve(),
        maximum_compiler_rows=args.maximum_compiler_rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualification"] != "NOT_QUALIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
