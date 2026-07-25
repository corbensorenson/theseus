#!/usr/bin/env python3
"""Issue an evidence-bound RDC/KERC resource disposition without rerunning MLX."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import host_resource_safety


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "rdc_kerc_resource_disposition.json"
POLICY = "project_theseus_rdc_kerc_resource_disposition_v1"
SOURCE_PATHS = (
    "reports/kerc_training_memory_preflight.json",
    "reports/kerc_training_memory_preflight.long_row.json",
    "reports/kerc_training_memory_preflight.long_row_cross_attention.json",
    "reports/kerc_training_memory_preflight.long_row_cross_attention_chunk512.json",
)


class ResourceDispositionFault(ValueError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResourceDispositionFault(f"json_object_required:{path}")
    return value


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def build_report() -> dict[str, Any]:
    contract_path = ROOT / "configs" / "pretraining_architecture_candidates.json"
    contract = read_json(contract_path)
    safety_mapping = contract.get("host_safety_policy") or {}
    safety = host_resource_safety.policy_from_mapping(
        safety_mapping, maximum_wall_seconds=1800.0
    )
    reports = {value: read_json(ROOT / value) for value in SOURCE_PATHS}
    if any(
        report.get("policy")
        != "project_theseus_kerc_training_memory_preflight_v1"
        for report in reports.values()
    ):
        raise ResourceDispositionFault("memory_evidence_policy_mismatch")
    for source, report in reports.items():
        for key in (
            "public_training_rows",
            "public_evaluation_rows",
            "external_inference_calls",
            "fallback_template_router_tool_credit",
        ):
            if int(report.get(key, -1)) != 0:
                raise ResourceDispositionFault(
                    f"no_cheat_counter_nonzero:{source}:{key}"
                )
    normal = reports[SOURCE_PATHS[0]]
    long_panel = reports[SOURCE_PATHS[1]]
    cross = reports[SOURCE_PATHS[2]]
    chunked = reports[SOURCE_PATHS[3]]
    long_stations = long_panel.get("stations") or []
    identities = {
        (
            int(row.get("row_index") or -1),
            str(row.get("row_sha256") or ""),
            int(row.get("sequence_width") or 0),
        )
        for row in [*long_stations, cross, chunked]
    }
    if len(identities) != 1:
        raise ResourceDispositionFault("long_row_identity_mismatch")
    row_index, row_sha256, sequence_width = identities.pop()
    if row_index < 0 or not row_sha256 or sequence_width <= 0:
        raise ResourceDispositionFault("long_row_identity_incomplete")
    long_peak = max(
        float(value)
        for value in (long_panel.get("peak_memory_mib_by_station") or {}).values()
    )
    cross_peak = float(cross["mlx_peak_memory_bytes"]) / (1024**2)
    chunked_peak = float(chunked["mlx_peak_memory_bytes"]) / (1024**2)
    safe_limit = float(safety.max_process_memory_mib)
    checks = {
        "source_evidence_green": normal.get("trigger_state") == "GREEN"
        and long_panel.get("trigger_state") == "GREEN"
        and cross.get("trigger_state") == "GREEN"
        and chunked.get("trigger_state") == "GREEN",
        "long_row_identity_exact": True,
        "long_row_exceeds_current_host_limit": long_peak > safe_limit,
        "cross_attention_is_dominant": cross_peak >= long_peak * 0.95,
        "query_chunking_remains_unsafe": chunked_peak > safe_limit,
        "no_capability_credit": all(
            report.get("capability_credit") == "NONE_RESOURCE_PREFLIGHT_ONLY"
            for report in reports.values()
        ),
    }
    if not all(checks.values()):
        raise ResourceDispositionFault("resource_disposition_checks_failed")
    return {
        "policy": POLICY,
        "schema_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "trigger_state": "GREEN",
        "support_state": "exact_row_resource_diagnosis_no_rerun",
        "disposition": "RESOURCE_DEFERRED_ON_THIS_HOST",
        "candidate_id": "rdc_kerc_adequacy",
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "physical_memory_mib": round(
                host_resource_safety.physical_memory_mib(), 3
            ),
        },
        "safety_limit_mib": round(safe_limit, 3),
        "row_identity": {
            "row_index": row_index,
            "row_sha256": row_sha256,
            "sequence_width": sequence_width,
        },
        "measurements": {
            "normal_panel_peak_mib": max(
                float(value)
                for value in (
                    normal.get("peak_memory_mib_by_station") or {}
                ).values()
            ),
            "long_panel_peak_mib": round(long_peak, 3),
            "cross_attention_peak_mib": round(cross_peak, 3),
            "query_chunk_512_peak_mib": round(chunked_peak, 3),
            "query_chunk_512_reduction_fraction": round(
                1.0 - chunked_peak / max(cross_peak, 1e-9), 6
            ),
        },
        "checks": checks,
        "source_artifacts": {
            **{value: artifact(ROOT / value) for value in SOURCE_PATHS},
            "candidate_contract": artifact(contract_path),
        },
        "candidate_execution_authorized": False,
        "scientific_falsification_claimed": False,
        "capability_claimed": False,
        "production_checkpoint_mutations": 0,
        "public_training_rows": 0,
        "public_evaluation_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
        "claim_boundary": (
            "This disposition says only that the exact long-sequence KERC candidate "
            "is unsafe within this 16 GiB host envelope. It does not falsify RDC, "
            "KERC, compact attention, query chunking, or their utility."
        ),
        "reentry_condition": (
            "Re-enter only after a mathematically equivalent memory-bounded source "
            "attention implementation passes output/loss/gradient parity on small "
            "cases and an externally guarded representative-row preflight stays "
            "below the host safety envelope without swap growth."
        ),
        "selected_first_campaign_action": (
            "Retain the canonical surface-control path and keep RDC/KERC state "
            "registered but excluded from first-campaign optimizer exposure."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    report = build_report()
    output = resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "disposition": report["disposition"],
                "measurements": report["measurements"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
