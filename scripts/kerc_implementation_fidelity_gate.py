#!/usr/bin/env python3
"""Independently gate the KERC source-to-implementation claim boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kernel_english_protocol  # noqa: E402
import kerc_semantic_corpus  # noqa: E402
import report_evidence_store  # noqa: E402


DEFAULT_CONTRACT = ROOT / "configs" / "kerc_implementation_fidelity.json"
DEFAULT_OUT = ROOT / "reports" / "kerc_implementation_fidelity_gate.json"
ALLOWED_STATUSES = {
    "faithful",
    "stronger_registered_alternative",
    "approximate",
    "fixture_only",
    "inactive",
    "absent",
}
REQUIRED_MECHANISM_FIELDS = {
    "id",
    "paper_sections",
    "status",
    "implementation_refs",
    "evidence_refs",
    "current_finding",
    "claim_ceiling",
    "next_evidence",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    report = audit_contract(read_json(resolve(args.contract)), contract_path=resolve(args.contract))
    report_evidence_store.write_json_report(resolve(args.out), report)
    print(json.dumps(gate_view(report) if args.gate else report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit_contract(
    contract: dict[str, Any],
    *,
    contract_path: Path | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    faults: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    path = contract_path or root / "configs" / "kerc_implementation_fidelity.json"
    if contract.get("policy") != "project_theseus_kerc_implementation_fidelity_v1":
        faults.append(fault("wrong_policy", observed=contract.get("policy")))

    configured_statuses = set(string_list(contract.get("allowed_statuses")))
    if configured_statuses != ALLOWED_STATUSES:
        faults.append(
            fault(
                "allowed_status_set_mismatch",
                observed=sorted(configured_statuses),
                required=sorted(ALLOWED_STATUSES),
            )
        )

    source_report = audit_sources(contract, root=root)
    faults.extend(source_report["faults"])
    corpus_path = source_report.get("corpus_path")
    corpus_report = audit_corpus(corpus_path) if isinstance(corpus_path, Path) and corpus_path.exists() else empty_corpus_report()
    faults.extend(corpus_report["faults"])

    expected_facts = contract.get("observed_corpus_contract")
    if not isinstance(expected_facts, dict):
        faults.append(fault("observed_corpus_contract_missing"))
        expected_facts = {}
    for key, expected in sorted(expected_facts.items()):
        observed = corpus_report["observed"].get(key)
        if observed != expected:
            faults.append(fault("corpus_fact_mismatch", field=key, expected=expected, observed=observed))

    mechanism_report = audit_mechanisms(contract, root=root, corpus_report=corpus_report)
    faults.extend(mechanism_report["faults"])
    hypothesis_report = audit_hypotheses(contract)
    faults.extend(hypothesis_report["faults"])
    behavior_report = audit_behavioral_claim_probes(corpus_report.get("sample_record"))
    faults.extend(behavior_report["faults"])

    claim_state = contract.get("claim_state") if isinstance(contract.get("claim_state"), dict) else {}
    required_claim_state = {
        "construct_validity": "RED",
        "learned_kerc": "NOT_CLAIMED",
        "utility": "NOT_CLAIMED",
        "efficiency": "NOT_CLAIMED",
        "scientific_negative": "NOT_CLAIMED",
        "required_failure_label_before_k7": "INCONCLUSIVE_IMPLEMENTATION",
    }
    for key, required in required_claim_state.items():
        if claim_state.get(key) != required:
            faults.append(fault("claim_state_overreach", field=key, required=required, observed=claim_state.get(key)))

    trigger_state = "GREEN" if not faults else "RED"
    return {
        "policy": "project_theseus_kerc_implementation_fidelity_gate_v1",
        "trigger_state": trigger_state,
        "contract": {
            "path": relative(path, root),
            "sha256": sha256_file(path) if path.exists() else None,
            "policy": contract.get("policy"),
        },
        "claim_state": claim_state,
        "summary": {
            "source_artifact_count": source_report["source_artifact_count"],
            "source_artifact_valid_count": source_report["source_artifact_valid_count"],
            "record_count": corpus_report["observed"]["record_count"],
            "mechanism_count": mechanism_report["mechanism_count"],
            "faithful_mechanism_count": mechanism_report["status_counts"].get("faithful", 0),
            "approximate_or_fixture_mechanism_count": sum(
                mechanism_report["status_counts"].get(key, 0) for key in ("approximate", "fixture_only")
            ),
            "inactive_or_absent_mechanism_count": sum(
                mechanism_report["status_counts"].get(key, 0) for key in ("inactive", "absent")
            ),
            "hypothesis_evidence_active_count": hypothesis_report["active_count"],
            "byte_literal_mutation_rejected": behavior_report["byte_literal_mutation_rejected"],
            "interaction_label_depends_on_global_dictionary": behavior_report[
                "interaction_label_depends_on_global_dictionary"
            ],
            "fault_count": len(faults),
            "warning_count": len(warnings),
        },
        "source_audit": {
            key: value for key, value in source_report.items() if key != "corpus_path"
        },
        "corpus_audit": {key: value for key, value in corpus_report.items() if key != "sample_record"},
        "mechanism_audit": mechanism_report,
        "hypothesis_audit": hypothesis_report,
        "behavioral_claim_probes": behavior_report,
        "faults": faults,
        "warnings": warnings,
        "non_claims": [
            "This GREEN gate means the K0 fidelity map is honest and source-bound, not that KERC is faithful or useful.",
            "The current presence-label head is not a learned per-unit residual allocator.",
            "Packet-level structural allocation does not establish semantic rate-distortion utility.",
            "No H1-H8 hypothesis is active before the preregistered matched campaign.",
            "The current corpus and verifier cannot support a scientific KERC failure verdict.",
        ],
    }


def audit_sources(contract: dict[str, Any], *, root: Path) -> dict[str, Any]:
    faults: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    declared = contract.get("source_artifacts") if isinstance(contract.get("source_artifacts"), list) else []
    paper = contract.get("paper") if isinstance(contract.get("paper"), dict) else {}
    declared = [{"path": paper.get("path"), "sha256": paper.get("sha256"), "kind": "paper"}, *declared]
    corpus_path: Path | None = None
    for item in declared:
        if not isinstance(item, dict):
            faults.append(fault("source_artifact_row_invalid", observed=repr(item)))
            continue
        raw_path = str(item.get("path") or "")
        candidate = resolve(raw_path, root=root)
        expected = normalize_sha(item.get("sha256"))
        observed = sha256_file(candidate) if candidate.exists() and candidate.is_file() else None
        valid = bool(raw_path and expected and observed == expected)
        rows.append(
            {
                "path": relative(candidate, root),
                "kind": item.get("kind") or "implementation_or_evidence",
                "exists": candidate.exists(),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "valid": valid,
            }
        )
        if not valid:
            faults.append(
                fault(
                    "source_artifact_stale_or_missing",
                    path=relative(candidate, root),
                    expected=expected,
                    observed=observed,
                )
            )
        if raw_path.endswith("candidate_records.jsonl"):
            corpus_path = candidate
    return {
        "source_artifact_count": len(rows),
        "source_artifact_valid_count": sum(1 for row in rows if row["valid"]),
        "sources": rows,
        "corpus_path": corpus_path,
        "faults": faults,
    }


def audit_corpus(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    dispositions: Counter[str] = Counter()
    sample_record: dict[str, Any] | None = None
    faults: list[dict[str, Any]] = []
    try:
        for record in jsonl_rows(path):
            if sample_record is None:
                sample_record = record
            counts["record_count"] += 1
            packet = record.get("kernel_packet") if isinstance(record.get("kernel_packet"), dict) else {}
            program = packet.get("program") if isinstance(packet.get("program"), dict) else {}
            nodes = program.get("nodes") if isinstance(program.get("nodes"), list) else []
            roots = program.get("roots") if isinstance(program.get("roots"), list) else []
            counts["multi_node_program_count"] += int(len(nodes) > 1)
            counts["multi_root_program_count"] += int(len(roots) > 1)
            counts["non_preserved_derivation_count"] += sum(
                1 for node in nodes if isinstance(node, dict) and node.get("derivation") != "preserved"
            )
            corrections = packet.get("correction_lattice") if isinstance(packet.get("correction_lattice"), dict) else {}
            counts["correction_alternative_count"] += len(
                corrections.get("corrections") if isinstance(corrections.get("corrections"), list) else []
            )
            concepts = packet.get("concept_capsules") if isinstance(packet.get("concept_capsules"), dict) else {}
            counts["concept_capsule_count"] += len(concepts)
            macro_registry = packet.get("macro_registry")
            counts["nonempty_macro_registry_count"] += int(nonempty(macro_registry))

            answer = record.get("answer_packet") if isinstance(record.get("answer_packet"), dict) else {}
            claims = answer.get("claims") if isinstance(answer.get("claims"), list) else []
            counts["multi_claim_answer_count"] += int(len(claims) > 1)
            counts["byte_literal_value_count"] += count_typed_values(answer, "byte_literal")
            decision = answer.get("decision") if isinstance(answer.get("decision"), dict) else {}
            disposition = str(decision.get("disposition") or "MISSING")
            dispositions[disposition] += 1
            counts["partial_disposition_count"] += int(disposition == "PARTIAL")

            hrl = record.get("hrl_state") if isinstance(record.get("hrl_state"), dict) else {}
            global_state = hrl.get("global") if isinstance(hrl.get("global"), dict) else {}
            global_payload = {key: value for key, value in global_state.items() if key != "language"}
            global_nonempty = nonempty(global_payload)
            counts["nonempty_interaction_global_dictionary_count"] += int(global_nonempty)
            supervision = record.get("residual_supervision") if isinstance(record.get("residual_supervision"), dict) else {}
            labels = supervision.get("labels_by_channel") if isinstance(supervision.get("labels_by_channel"), dict) else {}
            counts["interaction_presence_label_with_empty_global_count"] += int(
                int(labels.get("interaction") or 0) > 0 and not global_nonempty
            )
            allocation = (
                supervision.get("rate_distortion_allocation")
                if isinstance(supervision.get("rate_distortion_allocation"), dict)
                else {}
            )
            unit_rows = allocation.get("unit_allocations") if isinstance(allocation.get("unit_allocations"), list) else []
            counts["per_unit_allocation_receipt_count"] += len(unit_rows)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        faults.append(fault("corpus_scan_failed", path=str(path), error=f"{type(exc).__name__}:{exc}"))
    observed = {
        key: counts[key]
        for key in (
            "record_count",
            "multi_node_program_count",
            "multi_root_program_count",
            "non_preserved_derivation_count",
            "multi_claim_answer_count",
            "byte_literal_value_count",
            "correction_alternative_count",
            "concept_capsule_count",
            "nonempty_macro_registry_count",
            "nonempty_interaction_global_dictionary_count",
            "partial_disposition_count",
            "per_unit_allocation_receipt_count",
            "interaction_presence_label_with_empty_global_count",
        )
    }
    return {
        "path": relative(path, ROOT),
        "sha256": sha256_file(path) if path.exists() else None,
        "observed": observed,
        "disposition_counts": dict(sorted(dispositions.items())),
        "faults": faults,
        "sample_record": sample_record,
    }


def audit_mechanisms(
    contract: dict[str, Any],
    *,
    root: Path,
    corpus_report: dict[str, Any],
) -> dict[str, Any]:
    faults: list[dict[str, Any]] = []
    rows = [row for row in contract.get("mechanisms", []) if isinstance(row, dict)]
    required = set(string_list(contract.get("required_mechanism_ids")))
    observed_ids = [str(row.get("id") or "") for row in rows]
    duplicate_ids = sorted(identifier for identifier, count in Counter(observed_ids).items() if identifier and count > 1)
    missing = sorted(required - set(observed_ids))
    extra = sorted(set(observed_ids) - required)
    if duplicate_ids:
        faults.append(fault("duplicate_mechanism_ids", ids=duplicate_ids))
    if missing:
        faults.append(fault("missing_required_mechanisms", ids=missing))
    if extra:
        faults.append(fault("unexpected_mechanisms", ids=extra))
    status_counts: Counter[str] = Counter()
    by_id = {str(row.get("id") or ""): row for row in rows}
    for row in rows:
        identifier = str(row.get("id") or "")
        missing_fields = sorted(field for field in REQUIRED_MECHANISM_FIELDS if field not in row)
        if missing_fields:
            faults.append(fault("mechanism_fields_missing", mechanism_id=identifier, fields=missing_fields))
        status = str(row.get("status") or "")
        status_counts[status] += 1
        if status not in ALLOWED_STATUSES:
            faults.append(fault("mechanism_status_invalid", mechanism_id=identifier, status=status))
        for ref_kind in ("implementation_refs", "evidence_refs"):
            for ref in string_list(row.get(ref_kind)):
                if not resolve(ref, root=root).exists():
                    faults.append(fault("mechanism_reference_missing", mechanism_id=identifier, kind=ref_kind, path=ref))
        if not str(row.get("claim_ceiling") or "").strip():
            faults.append(fault("mechanism_claim_ceiling_missing", mechanism_id=identifier))

    facts = corpus_report.get("observed") if isinstance(corpus_report.get("observed"), dict) else {}
    per_unit = by_id.get("kerc.learned_per_unit_allocator", {})
    if facts.get("per_unit_allocation_receipt_count") == 0 and per_unit.get("status") != "absent":
        faults.append(fault("per_unit_allocator_overclaimed", status=per_unit.get("status")))
    interaction = by_id.get("kerc.interaction_amortization", {})
    if facts.get("nonempty_interaction_global_dictionary_count") == 0 and interaction.get("status") not in {"inactive", "absent"}:
        faults.append(fault("interaction_amortization_overclaimed", status=interaction.get("status")))
    macros = by_id.get("kerc.grammar_aware_macros", {})
    if facts.get("nonempty_macro_registry_count") == 0 and macros.get("status") != "absent":
        faults.append(fault("grammar_macro_overclaimed", status=macros.get("status")))
    compiler = by_id.get("kerc.surface_to_kernel_compiler", {})
    if facts.get("multi_node_program_count") == 0 and compiler.get("status") in {"faithful", "stronger_registered_alternative"}:
        faults.append(fault("semantic_compiler_overclaimed", status=compiler.get("status")))
    verifier = by_id.get("kerc.independent_roundtrip_verifier", {})
    if facts.get("byte_literal_value_count", 0) > 0 and verifier.get("status") in {"faithful", "stronger_registered_alternative"}:
        faults.append(fault("roundtrip_verifier_overclaimed", status=verifier.get("status")))
    return {
        "mechanism_count": len(rows),
        "required_mechanism_count": len(required),
        "status_counts": dict(sorted(status_counts.items())),
        "mechanisms": rows,
        "faults": faults,
    }


def audit_hypotheses(contract: dict[str, Any]) -> dict[str, Any]:
    faults: list[dict[str, Any]] = []
    rows = [row for row in contract.get("hypotheses", []) if isinstance(row, dict)]
    required = set(string_list(contract.get("required_hypothesis_ids")))
    observed = [str(row.get("id") or "") for row in rows]
    if set(observed) != required:
        faults.append(
            fault(
                "hypothesis_inventory_mismatch",
                missing=sorted(required - set(observed)),
                extra=sorted(set(observed) - required),
            )
        )
    if len(observed) != len(set(observed)):
        faults.append(fault("duplicate_hypothesis_ids", ids=observed))
    active_count = 0
    for row in rows:
        status = str(row.get("status") or "")
        if status not in {"inactive", "absent"}:
            active_count += 1
            faults.append(fault("hypothesis_evidence_activated_before_k8", hypothesis_id=row.get("id"), status=status))
        if not str(row.get("claim_ceiling") or "").strip():
            faults.append(fault("hypothesis_claim_ceiling_missing", hypothesis_id=row.get("id")))
    return {"hypothesis_count": len(rows), "active_count": active_count, "hypotheses": rows, "faults": faults}


def audit_behavioral_claim_probes(sample_record: dict[str, Any] | None) -> dict[str, Any]:
    faults: list[dict[str, Any]] = []
    residual = {
        "fidelity": "semantic",
        "segment_frame": {},
        "token_tags": {},
        "exact_object_handles": [],
    }
    empty = kerc_semantic_corpus.residual_supervision(
        "k0-empty",
        packet={"residual": residual},
        hrl_state={"segments": {}, "global": {}},
    )
    segment_only = kerc_semantic_corpus.residual_supervision(
        "k0-segment-only",
        packet={"residual": residual},
        hrl_state={"segments": {"turn": {"entries": {"x": 1}}}, "global": {}},
    )
    interaction_depends_on_global = not (
        empty["labels_by_channel"]["interaction"] == 0
        and segment_only["labels_by_channel"]["interaction"] == 1
    )
    if interaction_depends_on_global:
        faults.append(fault("interaction_label_probe_changed_without_contract_update"))

    byte_literal_mutation_rejected: bool | None = None
    if isinstance(sample_record, dict):
        answer = sample_record.get("answer_packet")
        packet = sample_record.get("kernel_packet") if isinstance(sample_record.get("kernel_packet"), dict) else {}
        protected = packet.get("protected_objects") if isinstance(packet.get("protected_objects"), dict) else {}
        if isinstance(answer, dict):
            changed = copy.deepcopy(answer)
            mutated = mutate_first_byte_literal(changed)
            if not mutated:
                faults.append(fault("byte_literal_probe_sample_missing"))
            else:
                result = kernel_english_protocol.verify_answer_roundtrip(answer, changed, protected_objects=protected)
                byte_literal_mutation_rejected = not bool(result.get("passes"))
                if byte_literal_mutation_rejected:
                    faults.append(
                        fault(
                            "byte_literal_probe_contract_changed",
                            reason="The fidelity map classifies the current verifier as approximate because this mutation is accepted; refresh K0 if the verifier was repaired.",
                        )
                    )
    else:
        faults.append(fault("behavior_probe_sample_missing"))
    return {
        "interaction_label_empty_segments": empty["labels_by_channel"]["interaction"],
        "interaction_label_segment_state_only": segment_only["labels_by_channel"]["interaction"],
        "interaction_label_depends_on_global_dictionary": interaction_depends_on_global,
        "byte_literal_mutation_rejected": byte_literal_mutation_rejected,
        "faults": faults,
    }


def mutate_first_byte_literal(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "byte_literal" and isinstance(value.get("value"), str):
            value["value"] = value["value"] + "QQ=="
            return True
        return any(mutate_first_byte_literal(item) for item in value.values())
    if isinstance(value, list):
        return any(mutate_first_byte_literal(item) for item in value)
    return False


def count_typed_values(value: Any, type_name: str) -> int:
    if isinstance(value, dict):
        return int(value.get("type") == type_name) + sum(count_typed_values(item, type_name) for item in value.values())
    if isinstance(value, list):
        return sum(count_typed_values(item, type_name) for item in value)
    return 0


def nonempty(value: Any) -> bool:
    if isinstance(value, dict):
        return any(nonempty(item) for item in value.values())
    if isinstance(value, list):
        return any(nonempty(item) for item in value)
    return value not in (None, "", False, 0)


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"row {line_number} is not an object")
            yield row


def empty_corpus_report() -> dict[str, Any]:
    return {
        "path": None,
        "sha256": None,
        "observed": {key: 0 for key in (
            "record_count",
            "multi_node_program_count",
            "multi_root_program_count",
            "non_preserved_derivation_count",
            "multi_claim_answer_count",
            "byte_literal_value_count",
            "correction_alternative_count",
            "concept_capsule_count",
            "nonempty_macro_registry_count",
            "nonempty_interaction_global_dictionary_count",
            "partial_disposition_count",
            "per_unit_allocation_receipt_count",
            "interaction_presence_label_with_empty_global_count",
        )},
        "disposition_counts": {},
        "faults": [fault("corpus_missing")],
        "sample_record": None,
    }


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report["trigger_state"],
        **report["summary"],
        "claim_state": report["claim_state"],
        "faults": report["faults"][:20],
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def resolve(raw: str, *, root: Path = ROOT) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def normalize_sha(value: Any) -> str:
    return str(value or "").removeprefix("sha256:")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def fault(kind: str, **evidence: Any) -> dict[str, Any]:
    return {"kind": kind, "evidence": evidence}


if __name__ == "__main__":
    raise SystemExit(main())
