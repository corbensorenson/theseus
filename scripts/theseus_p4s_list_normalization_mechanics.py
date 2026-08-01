#!/usr/bin/env python3
"""Audit v2r2 list mechanics on retained P4S text without rescoring candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import theseus_semantic_ir_v2 as v2
import theseus_semantic_ir_v2r2 as v2r2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "theseus_p4s_list_normalization_mechanics.json"
RUNTIME_GLOB = (
    "p2a_p4_p4s-cognitive-compilation-*_typed_semantic_ir_treatment_"
    "attempt2_direct_local_model_*.json"
)
REPLACEMENT_RE = re.compile(r"<<<\n(.*?)\n>>>", flags=re.DOTALL)


def build_report() -> dict[str, Any]:
    files = sorted((ROOT / "runtime" / "p2a").glob(RUNTIME_GLOB))
    rows: list[dict[str, Any]] = []
    surface_counts: Counter[str] = Counter()
    for path in files:
        payload = read_json(path)
        answer = str((payload.get("response") or {}).get("answer") or "")
        normalized, receipt = v2r2.normalize_with_receipt(answer)
        before_payload_hashes = replacement_hashes(answer)
        after_payload_hashes = replacement_hashes(normalized)
        classes = [
            str(field.get("surface_class") or "")
            for field in receipt["fields"]
            if field.get("surface_class")
        ]
        surface_counts.update(classes)
        faults: list[str] = []
        if not answer:
            faults.append("retained_answer_missing")
        if receipt["rejected_field_count"]:
            faults.append("list_surface_rejected")
        if not list_fields_are_canonical(normalized):
            faults.append("normalized_list_fields_not_canonical")
        if before_payload_hashes != after_payload_hashes:
            faults.append("replacement_payload_changed")
        if non_list_projection(answer) != non_list_projection(normalized):
            faults.append("non_list_surface_changed")
        rows.append({
            "runtime_receipt": relative(path),
            "runtime_receipt_sha256": file_sha256(path),
            "retained_answer_sha256": sha256_text(answer),
            "normalized_transport_sha256": sha256_text(normalized),
            "v2_syntax_complete_before_normalization": v2.complete(answer),
            "v2_syntax_complete_after_normalization": v2.complete(normalized),
            "list_fields_canonical_after_normalization": list_fields_are_canonical(
                normalized
            ),
            "list_field_count": receipt["list_field_count"],
            "normalized_field_count": receipt["normalized_field_count"],
            "surface_classes": classes,
            "replacement_payload_sha256": before_payload_hashes,
            "non_list_surface_preserved": non_list_projection(answer)
            == non_list_projection(normalized),
            "faults": faults,
            "state": "GREEN" if not faults else "RED",
        })
    faults: list[str] = []
    if len(rows) != 20:
        faults.append(f"retained_attempt2_treatment_call_count:{len(rows)}")
    if any(row["state"] != "GREEN" for row in rows):
        faults.append("retained_list_mechanics_not_green")
    disposition = ROOT / "reports" / "theseus_p4s_terminal_disposition.json"
    return {
        "policy": "project_theseus_p4s_list_normalization_mechanics_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": faults,
        "prospective_parser": "scripts/theseus_semantic_ir_v2r2.py",
        "prospective_parser_sha256": file_sha256(
            ROOT / "scripts" / "theseus_semantic_ir_v2r2.py"
        ),
        "retained_p4s_terminal_disposition": relative(disposition),
        "retained_p4s_terminal_disposition_sha256": file_sha256(disposition),
        "retained_attempt2_treatment_call_count": len(rows),
        "surface_class_counts": dict(sorted(surface_counts.items())),
        "unrelated_non_list_syntax_residual_count": sum(
            not row["v2_syntax_complete_after_normalization"] for row in rows
        ),
        "rows": rows,
        "custody": {
            "mechanics_only": True,
            "candidate_correctness_evaluated": False,
            "task_evaluator_invocations": 0,
            "candidate_actions_applied": 0,
            "p4s_scores_recomputed": False,
            "p4s_disposition_modified": False,
            "consumed_runtime_receipts_modified": 0,
            "identifier_values_invented": 0,
            "identifier_order_changed": False,
            "replacement_source_changed": False,
        },
        "maximum_inference": (
            "The prospective v2r2 parser can canonicalize the observed delimiter-only "
            "list spellings while preserving emitted identifier order and replacement "
            "bytes. Two retained calls still have an unrelated overlong NODE field; "
            "list normalization does not repair or hide that residual. This is mechanics "
            "evidence only. It does not rescore P4S, establish "
            "candidate correctness, show a treatment effect, qualify D1, support the "
            "book claim, or replace a fresh source-disjoint P4 denominator."
        ),
    }


def replacement_hashes(text: str) -> list[str]:
    return [sha256_text(match) for match in REPLACEMENT_RE.findall(text)]


def non_list_projection(text: str) -> str:
    lines: list[str] = []
    inside_replacement = False
    for line in v2.unwrap(text).splitlines():
        if line == "<<<" and not inside_replacement:
            inside_replacement = True
        elif line == ">>>" and inside_replacement:
            inside_replacement = False
        if not inside_replacement and re.fullmatch(
            r"(?:ALL_OBLIGATIONS|OBLIGATIONS|LOSS) .+", line
        ):
            lines.append(line.split(" ", 1)[0] + " <LIST>")
        else:
            lines.append(line)
    return "\n".join(lines)


def list_fields_are_canonical(text: str) -> bool:
    fields = []
    inside_replacement = False
    for line in v2.unwrap(text).splitlines():
        if line == "<<<" and not inside_replacement:
            inside_replacement = True
            continue
        if line == ">>>" and inside_replacement:
            inside_replacement = False
            continue
        if inside_replacement:
            continue
        match = re.fullmatch(r"(ALL_OBLIGATIONS|OBLIGATIONS|LOSS) (.+)", line)
        if match is not None:
            field, surface = match.groups()
            fields.append(
                (field == "LOSS" and surface == "NONE")
                or bool(re.fullmatch(r"[A-Z][A-Z0-9_]*(?:,[A-Z][A-Z0-9_]*)*", surface))
            )
    return bool(fields) and all(fields)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if not args.audit_only:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "faults": report["faults"],
        "retained_attempt2_treatment_call_count": report[
            "retained_attempt2_treatment_call_count"
        ],
        "surface_class_counts": report["surface_class_counts"],
    }, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
