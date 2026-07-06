#!/usr/bin/env python3
"""Bind generator, integrity, and verifier obligations to semantic IR records.

This gate consumes the materialized VIEA semantic atom/node view and emits
explicit obligation/dependency/evidence-binding records for the current
candidate-integrity, private-verifier, and direct-generator boundaries. It does
not run model generation, verification, or public calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402


DEFAULT_VIEW = ROOT / "reports" / "viea_spine_materialized_view.json"
DEFAULT_CANDIDATE = ROOT / "reports" / "candidate_integrity_audit.json"
DEFAULT_VERIFIER = ROOT / "reports" / "private_verifier_spine_smoke.json"
DEFAULT_GENERATOR = ROOT / "reports" / "neural_seed_token_decoder_comparator.json"
DEFAULT_OUT = ROOT / "reports" / "semantic_ir_obligation_gate.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "semantic_ir_obligation_gate.md"
NO_CHEAT = {
    "public_training_rows_written": 0,
    "external_inference_calls": 0,
    "fallback_return_count": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", default=rel(DEFAULT_VIEW))
    parser.add_argument("--candidate-integrity", default=rel(DEFAULT_CANDIDATE))
    parser.add_argument("--private-verifier", default=rel(DEFAULT_VERIFIER))
    parser.add_argument("--direct-generator", default=rel(DEFAULT_GENERATOR))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started)
    report_evidence_store.write_json_report(
        resolve(args.out),
        report,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(report),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    view_path = resolve(args.view)
    candidate_path = resolve(args.candidate_integrity)
    verifier_path = resolve(args.private_verifier)
    generator_path = resolve(args.direct_generator)
    view = read_json(view_path)
    candidate = read_json(candidate_path)
    verifier = read_json(verifier_path)
    generator = read_json(generator_path)

    semantic_records = list_dicts(view.get("semantic_ir_records"))
    semantic_atoms = [row for row in semantic_records if row.get("canonical_record_type") == "semantic_atom"]
    semantic_nodes = [row for row in semantic_records if row.get("canonical_record_type") == "semantic_node"]
    semantic_ready = (
        view.get("trigger_state") == "GREEN"
        and int_value(object_field(view.get("summary")).get("schema_payload_gap_count")) == 0
        and len(semantic_atoms) > 0
        and len(semantic_nodes) > 0
    )
    consumers = [
        consumer_state(
            "candidate_integrity",
            candidate_path,
            candidate,
            required_policy="project_theseus_candidate_integrity_audit_v1",
            allowed_states={"GREEN"},
            obligation_kind="candidate_family_integrity_and_generation_credit",
        ),
        consumer_state(
            "private_verifier",
            verifier_path,
            verifier,
            required_policy="project_theseus_private_verifier_spine_smoke_v1",
            allowed_states={"GREEN"},
            obligation_kind="runtime_load_and_behavior_label_verification",
        ),
        consumer_state(
            "direct_generator",
            generator_path,
            generator,
            required_policy="project_theseus_neural_seed_token_decoder_comparator_report_v0",
            allowed_states={"PLANNED", "GREEN", "YELLOW"},
            obligation_kind="prompt_signature_direct_generation_boundary",
        ),
    ]
    ready_consumers = [row for row in consumers if row["ready"]]
    hard_gates = [
        gate("materialized_semantic_view_ready", semantic_ready, {"path": rel(view_path), "semantic_atom_count": len(semantic_atoms), "semantic_node_count": len(semantic_nodes)}),
        gate("candidate_integrity_consumer_ready", consumers[0]["ready"], consumers[0]),
        gate("private_verifier_consumer_ready", consumers[1]["ready"], consumers[1]),
        gate("direct_generator_consumer_present", consumers[2]["ready"], consumers[2]),
    ]
    hard_failed = [row for row in hard_gates if not row["passed"]]
    records = build_records(
        view_path=view_path,
        semantic_atoms=semantic_atoms,
        semantic_nodes=semantic_nodes,
        consumers=consumers,
        ready_consumers=ready_consumers,
        support_state="SUPPORTED" if not hard_failed else "UNSUPPORTED",
    )
    return {
        "policy": "project_theseus_semantic_ir_obligation_gate_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_failed else "RED",
        "summary": {
            "materialized_view": rel(view_path),
            "semantic_record_count": len(semantic_records),
            "semantic_atom_count": len(semantic_atoms),
            "semantic_node_count": len(semantic_nodes),
            "consumer_count": len(consumers),
            "ready_consumer_count": len(ready_consumers),
            "semantic_obligation_record_count": len(records["semantic_obligation_records"]),
            "dependency_edge_record_count": len(records["dependency_edge_records"]),
            "evidence_binding_record_count": len(records["evidence_binding_records"]),
            "hard_failed_gate_count": len(hard_failed),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "hard_gates": hard_gates,
        "hard_gaps": [gap("semantic_obligation_binding_failed", hard_failed)] if hard_failed else [],
        "semantic_source_receipt": {
            "record_type": "semantic_source_receipt",
            "receipt_id": stable_id("semantic_source_receipt", rel(view_path), len(semantic_atoms), len(semantic_nodes)),
            "view_path": rel(view_path),
            "view_trigger_state": view.get("trigger_state"),
            "view_content_hash": stable_hash(view),
            "semantic_atom_count": len(semantic_atoms),
            "semantic_node_count": len(semantic_nodes),
            "ready": semantic_ready,
            **NO_CHEAT,
            "non_claims": ["semantic source receipt only", "not generation or verification execution"],
        },
        "consumer_states": consumers,
        **records,
        **NO_CHEAT,
        "non_claims": [
            "This gate binds obligations to semantic IR; it does not run model decoding.",
            "Semantic IR binding is not a public benchmark result or learned-generation promotion.",
            "Router/tool/template behavior remains separate from learned generation.",
        ],
    }


def consumer_state(
    surface: str,
    path: Path,
    payload: dict[str, Any],
    *,
    required_policy: str,
    allowed_states: set[str],
    obligation_kind: str,
) -> dict[str, Any]:
    summary = object_field(payload.get("summary"))
    return {
        "consumer_surface": surface,
        "path": rel(path),
        "content_hash": stable_hash(payload),
        "policy": payload.get("policy"),
        "trigger_state": payload.get("trigger_state"),
        "required_policy": required_policy,
        "allowed_states": sorted(allowed_states),
        "obligation_kind": obligation_kind,
        "candidate_count": summary.get("candidate_count"),
        "candidate_attempt_count": summary.get("candidate_attempt_count"),
        "integrity_verified_candidate_count": summary.get("integrity_verified_candidate_count"),
        "runtime_load_rate": summary.get("runtime_load_rate"),
        "intended_behavior_pass_rate": summary.get("intended_behavior_pass_rate"),
        "vcm_context_ready": first_present(summary, "vcm_context_governor_ready", "direct_generator_vcm_context_ready"),
        "viea_spine_view_ready": summary.get("viea_spine_view_ready"),
        "external_inference_calls": int_value(payload.get("external_inference_calls")),
        "ready": path.exists()
        and payload.get("policy") == required_policy
        and payload.get("trigger_state") in allowed_states
        and int_value(payload.get("external_inference_calls")) == 0,
    }


def build_records(
    *,
    view_path: Path,
    semantic_atoms: list[dict[str, Any]],
    semantic_nodes: list[dict[str, Any]],
    consumers: list[dict[str, Any]],
    ready_consumers: list[dict[str, Any]],
    support_state: str,
) -> dict[str, list[dict[str, Any]]]:
    atom_refs = compact_refs(semantic_atoms, limit=6)
    node_refs = compact_refs(semantic_nodes, limit=6)
    semantic_atom_records = [
        {
            "record_type": "semantic_atom",
            "record_id": f"semantic-obligation-atom-{index}",
            "target": consumer["consumer_surface"],
            "semantic_hash": stable_id("semantic_atom", consumer["consumer_surface"], consumer["obligation_kind"]),
            "support_state": "semantic_ir_bound",
            "source_refs": [consumer["path"], rel(view_path)],
            "evidence_refs": [consumer["path"], "reports/semantic_ir_obligation_gate.json"],
            **NO_CHEAT,
            "non_claims": ["consumer obligation atom", "not model capability"],
        }
        for index, consumer in enumerate(ready_consumers)
    ]
    semantic_node_records = [
        {
            "record_type": "semantic_node",
            "record_id": f"semantic-obligation-node-{index}",
            "node_id": f"semantic_obligation.{consumer['consumer_surface']}",
            "target": consumer["consumer_surface"],
            "source_refs": [consumer["path"], rel(view_path)],
            "evidence_refs": [consumer["path"], "reports/semantic_ir_obligation_gate.json"],
            **NO_CHEAT,
            "non_claims": ["consumer obligation node", "not model capability"],
        }
        for index, consumer in enumerate(ready_consumers)
    ]
    semantic_obligations = []
    dependency_edges = []
    evidence_bindings = []
    for consumer in ready_consumers:
        oid = stable_id("semantic_obligation", consumer["consumer_surface"], consumer["content_hash"], atom_refs, node_refs)
        semantic_obligations.append(
            {
                "record_type": "semantic_obligation",
                "record_id": f"semantic-obligation-{oid}",
                "consumer_surface": consumer["consumer_surface"],
                "obligation_kind": consumer["obligation_kind"],
                "support_state": support_state,
                "semantic_atom_refs": atom_refs,
                "semantic_node_refs": node_refs,
                "consumer_report_ref": consumer["path"],
                "required_behavior": "consumer must cite shared semantic atom/node vocabulary before promotion or routing claims use its obligations",
                **NO_CHEAT,
                "non_claims": ["semantic obligation binding only", "not benchmark or learned-generation evidence"],
            }
        )
        dependency_edges.append(
            {
                "record_type": "dependency_edge",
                "record_id": f"semantic-dependency-{oid}",
                "from_ref": rel(view_path),
                "to_ref": consumer["path"],
                "edge_kind": "semantic_ir_obligation_dependency",
                "support_state": support_state,
                **NO_CHEAT,
                "non_claims": ["dependency edge only"],
            }
        )
        evidence_bindings.append(
            {
                "record_type": "evidence_binding",
                "record_id": f"semantic-evidence-binding-{oid}",
                "consumer_surface": consumer["consumer_surface"],
                "evidence_refs": [consumer["path"], rel(view_path), "reports/semantic_ir_obligation_gate.json"],
                "semantic_atom_refs": atom_refs,
                "semantic_node_refs": node_refs,
                "support_state": support_state,
                **NO_CHEAT,
                "non_claims": ["evidence binding only"],
            }
        )
    claim_records = [
        {
            "record_type": "claim_record",
            "claim_id": "claim.semantic_ir_obligation_binding.v1",
            "claim": "Candidate-integrity, private-verifier, and direct-generator boundaries now bind obligations to the shared materialized semantic IR view.",
            "support_state": support_state,
            "evidence_refs": ["reports/semantic_ir_obligation_gate.json", rel(view_path)] + [row["path"] for row in ready_consumers],
            **NO_CHEAT,
            "non_claims": ["implementation cohesion claim", "not model capability"],
        }
    ]
    artifact_graph_records = [
        {
            "record_type": "artifact_graph_record",
            "artifact_id": "artifact.semantic_ir_obligation_gate.v1",
            "artifact_type": "semantic_ir_obligation_gate",
            "parent_job": "semantic_ir_obligation_gate",
            "source_refs": [rel(view_path)] + [row["path"] for row in consumers],
            "context_refs": [rel(view_path)],
            "context_transaction_refs": [],
            "semantic_certificate_refs": atom_refs + node_refs,
            "tool_refs": [],
            "claim_refs": [claim_records[0]["claim_id"]],
            "test_refs": ["python3 scripts/semantic_ir_obligation_gate.py"],
            "audit_events": ["materialized_semantic_view_loaded", "consumer_reports_loaded", "obligations_bound"],
            "replay_metadata": {"ready_consumer_count": len(ready_consumers), "semantic_atom_refs": atom_refs, "semantic_node_refs": node_refs},
            "replay_grade": "metadata_replayable_from_registered_reports",
            "environment_assumptions": ["local materialized VIEA view is current"],
            "provenance_status": "registered_semantic_obligation_binding",
            "replay_limits": ["does not execute generator", "does not execute verifier"],
            "evidence_gate": {"state": support_state, **NO_CHEAT},
            "residuals": [] if support_state == "SUPPORTED" else ["semantic obligation binding failed"],
            **NO_CHEAT,
            "non_claims": ["not a public calibration", "not a learned-generation claim"],
        }
    ]
    evidence_transition_records = [
        {
            "record_type": "evidence_transition_record",
            "record_id": "evidence.semantic_ir_obligation_binding.v1",
            "artifact_ref": "reports/semantic_ir_obligation_gate.json",
            "previous_support_state": "PLANNER_ONLY_SEMANTIC_IR_VIEW",
            "current_support_state": support_state,
            "transition_reason": "candidate/verifier/generator obligations consume materialized semantic atom and semantic node references",
            "evidence_ref": "reports/semantic_ir_obligation_gate.json",
            **NO_CHEAT,
            "non_claims": ["semantic IR evidence transition only"],
        }
    ]
    return {
        "semantic_atom_records": semantic_atom_records,
        "semantic_node_records": semantic_node_records,
        "semantic_obligation_records": semantic_obligations,
        "dependency_edge_records": dependency_edges,
        "evidence_binding_records": evidence_bindings,
        "claim_records": claim_records,
        "artifact_graph_records": artifact_graph_records,
        "evidence_transition_records": evidence_transition_records,
    }


def compact_refs(rows: list[dict[str, Any]], *, limit: int) -> list[str]:
    refs = []
    for row in rows[: max(1, limit)]:
        ref = str(row.get("record_id") or row.get("content_hash") or "")
        if ref:
            refs.append(ref)
    return refs


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def gap(kind: str, evidence: Any) -> dict[str, Any]:
    return {"kind": kind, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def object_field(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def stable_id(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except Exception:
        return str(value)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report.get("summary"))
    return "\n".join(
        [
            "# Semantic IR Obligation Gate",
            "",
            f"- Trigger state: `{report.get('trigger_state')}`",
            f"- Semantic atoms: `{summary.get('semantic_atom_count')}`",
            f"- Semantic nodes: `{summary.get('semantic_node_count')}`",
            f"- Ready consumers: `{summary.get('ready_consumer_count')}/{summary.get('consumer_count')}`",
            f"- Obligation records: `{summary.get('semantic_obligation_record_count')}`",
            "",
            "This report binds existing generator/verifier/integrity obligations to the shared semantic IR view. It does not execute model generation or public calibration.",
        ]
    ) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
