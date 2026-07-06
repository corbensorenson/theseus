#!/usr/bin/env python3
"""Audit semantic atoms and semantic patches for Theseus changes.

This gate makes a repo change reviewable as obligations over localized atoms
rather than as a vague whole-task success/failure. It does not apply patches;
it validates that patches carry source anchors, effect logs, validation
obligations, rollback notes, and explicit non-claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "semantic_patch_registry.json"
DEFAULT_REPORT = ROOT / "reports" / "semantic_patch_gate.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "semantic_patch_gate.md"
DEFAULT_ATOMS = ROOT / "reports" / "semantic_atoms.jsonl"
DEFAULT_PATCHES = ROOT / "reports" / "semantic_patches.jsonl"
DEFAULT_EFFECTS = ROOT / "reports" / "semantic_effect_logs.jsonl"

REQUIRED_ATOM_FIELDS = {
    "id",
    "type",
    "artifact",
    "source_anchor",
    "obligation_ids",
    "repair_scope",
    "authority_boundary",
}
REQUIRED_PATCH_FIELDS = {
    "id",
    "artifact",
    "affected_atoms",
    "obligations",
    "validation_commands",
    "side_effects",
    "rollback_notes",
    "effect_log_refs",
    "non_claims",
}
REQUIRED_EFFECT_FIELDS = {
    "id",
    "command",
    "inputs",
    "outputs",
    "allowed_effects",
    "forbidden_effects",
    "replay_status",
}
REQUIRED_FIXTURE_FIELDS = {
    "id",
    "broken_atom_id",
    "simulated_breakage",
    "repair_scope",
    "current_repair_evidence",
    "scope_change_ledger",
    "expected_scope_result",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--atoms-out", default=rel(DEFAULT_ATOMS))
    parser.add_argument("--patches-out", default=rel(DEFAULT_PATCHES))
    parser.add_argument("--effects-out", default=rel(DEFAULT_EFFECTS))
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = build_report(config_path, config, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.atoms_out), report["materialized_atoms"])
    write_jsonl(resolve(args.patches_out), report["materialized_patches"])
    write_jsonl(resolve(args.effects_out), report["materialized_effect_logs"])
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(config_path: Path, config: dict[str, Any], started: float) -> dict[str, Any]:
    atoms = [audit_atom(row) for row in list_dicts(config.get("semantic_atoms"))]
    atom_by_id = {row["id"]: row for row in atoms}
    effects = [audit_effect(row) for row in list_dicts(config.get("effect_logs"))]
    effect_by_id = {row["id"]: row for row in effects}
    patches = [audit_patch(row, atom_by_id, effect_by_id) for row in list_dicts(config.get("semantic_patches"))]
    fixtures = [audit_fixture(row, atom_by_id) for row in list_dicts(config.get("localized_repair_fixtures"))]
    boundary_gates = audit_boundaries(dict_value(config.get("hard_boundaries")))

    hard_gaps = [
        gap
        for collection in [atoms, effects, patches, fixtures]
        for row in collection
        for gap in row["hard_gaps"]
    ] + [gate for gate in boundary_gates if gate["severity"] == "hard" and not gate["passed"]]
    warnings = [
        warning
        for collection in [atoms, effects, patches, fixtures]
        for row in collection
        for warning in row["warnings"]
    ]
    trigger_state = "GREEN"
    if hard_gaps:
        trigger_state = "RED"
    elif warnings:
        trigger_state = "YELLOW"

    summary = {
        "config": rel(config_path),
        "atom_count": len(atoms),
        "patch_count": len(patches),
        "effect_log_count": len(effects),
        "fixture_count": len(fixtures),
        "anchored_atom_count": sum(1 for row in atoms if row["source_anchor_present"]),
        "patches_with_validation_commands": sum(1 for row in patches if row["validation_command_count"] > 0),
        "patches_with_effect_logs": sum(1 for row in patches if row["effect_log_count"] > 0),
        "fixtures_inside_scope": sum(1 for row in fixtures if row["inside_repair_scope"]),
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
    }
    return {
        "policy": "project_theseus_semantic_patch_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "boundary_gates": boundary_gates,
        "atoms": atoms,
        "patches": patches,
        "effect_logs": effects,
        "localized_repair_fixtures": fixtures,
        "materialized_atoms": [materialized_atom(row) for row in atoms],
        "materialized_patches": [materialized_patch(row) for row in patches],
        "materialized_effect_logs": [materialized_effect(row) for row in effects],
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "rules": {
            "semantic_atom": "A failed generator/repo obligation should point to a typed atom and source anchor, not only a whole task row.",
            "semantic_patch": "A patch is not complete until affected atoms, validation obligations, effects, rollback, and non-claims are explicit.",
            "scope": "A repair outside the atom repair scope must emit a scope-change ledger entry.",
            "syntax": "Syntactic patch success does not discharge semantic obligations.",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def audit_boundaries(boundaries: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        gate("syntactic_patch_success_not_enough", boundaries.get("syntactic_patch_success_discharges_obligations") is False, "hard", boundaries.get("syntactic_patch_success_discharges_obligations")),
        gate("unbounded_effects_forbidden", boundaries.get("unbounded_effects_allowed") is False, "hard", boundaries.get("unbounded_effects_allowed")),
        gate("replay_effect_log_required", boundaries.get("replay_effect_log_required") is True, "hard", boundaries.get("replay_effect_log_required")),
        gate("scope_change_requires_ledger", boundaries.get("scope_change_requires_ledger") is True, "hard", boundaries.get("scope_change_requires_ledger")),
        gate("public_benchmark_training_forbidden", boundaries.get("public_benchmark_training_allowed") is False, "hard", boundaries.get("public_benchmark_training_allowed")),
        gate("runtime_external_inference_forbidden", boundaries.get("runtime_external_inference_allowed") is False, "hard", boundaries.get("runtime_external_inference_allowed")),
    ]


def audit_atom(row: dict[str, Any]) -> dict[str, Any]:
    atom_id = str(row.get("id") or "<missing-id>")
    missing = sorted(REQUIRED_ATOM_FIELDS - set(row))
    artifact = resolve(str(row.get("artifact") or ""))
    anchor = str(row.get("source_anchor") or "")
    text = artifact.read_text(encoding="utf-8") if artifact.exists() and artifact.is_file() else ""
    anchor_present = bool(anchor and anchor in text)
    hard_gaps = []
    warnings = []
    if missing:
        hard_gaps.append(gap(atom_id, "missing_atom_fields", {"fields": missing}))
    if not artifact.exists():
        hard_gaps.append(gap(atom_id, "artifact_missing", {"artifact": str(row.get("artifact") or "")}))
    if not anchor_present:
        hard_gaps.append(gap(atom_id, "source_anchor_missing", {"artifact": str(row.get("artifact") or ""), "anchor": anchor}))
    if not list_values(row.get("obligation_ids")):
        hard_gaps.append(gap(atom_id, "obligations_missing", {}))
    if not list_values(row.get("repair_scope")):
        hard_gaps.append(gap(atom_id, "repair_scope_missing", {}))
    return {
        "id": atom_id,
        "type": str(row.get("type") or ""),
        "artifact": str(row.get("artifact") or ""),
        "artifact_sha256": file_hash(artifact) if artifact.exists() and artifact.is_file() else "",
        "source_anchor_present": anchor_present,
        "obligation_ids": list_values(row.get("obligation_ids")),
        "repair_scope": list_values(row.get("repair_scope")),
        "authority_boundary": str(row.get("authority_boundary") or ""),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def audit_effect(row: dict[str, Any]) -> dict[str, Any]:
    effect_id = str(row.get("id") or "<missing-id>")
    missing = sorted(REQUIRED_EFFECT_FIELDS - set(row))
    inputs = [resolve(str(path)) for path in list_values(row.get("inputs"))]
    outputs = [resolve(str(path)) for path in list_values(row.get("outputs"))]
    missing_inputs = [rel(path) for path in inputs if not path.exists()]
    missing_outputs = [rel(path) for path in outputs if not path.exists()]
    hard_gaps = []
    warnings = []
    if missing:
        hard_gaps.append(gap(effect_id, "missing_effect_fields", {"fields": missing}))
    if missing_inputs:
        hard_gaps.append(gap(effect_id, "effect_inputs_missing", {"missing": missing_inputs}))
    if missing_outputs:
        warnings.append(gap(effect_id, "effect_outputs_missing", {"missing": missing_outputs}, severity="warning"))
    if not list_values(row.get("forbidden_effects")):
        hard_gaps.append(gap(effect_id, "forbidden_effects_missing", {}))
    return {
        "id": effect_id,
        "command": str(row.get("command") or ""),
        "input_count": len(inputs),
        "output_count": len(outputs),
        "missing_inputs": missing_inputs,
        "missing_outputs": missing_outputs,
        "allowed_effects": list_values(row.get("allowed_effects")),
        "forbidden_effects": list_values(row.get("forbidden_effects")),
        "replay_status": str(row.get("replay_status") or ""),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def audit_patch(row: dict[str, Any], atoms: dict[str, dict[str, Any]], effects: dict[str, dict[str, Any]]) -> dict[str, Any]:
    patch_id = str(row.get("id") or "<missing-id>")
    missing = sorted(REQUIRED_PATCH_FIELDS - set(row))
    affected = [str(x) for x in list_values(row.get("affected_atoms"))]
    effect_refs = [str(x) for x in list_values(row.get("effect_log_refs"))]
    missing_atoms = [atom for atom in affected if atom not in atoms]
    missing_effects = [effect for effect in effect_refs if effect not in effects]
    hard_gaps = []
    warnings = []
    if missing:
        hard_gaps.append(gap(patch_id, "missing_patch_fields", {"fields": missing}))
    if missing_atoms:
        hard_gaps.append(gap(patch_id, "affected_atoms_missing", {"missing": missing_atoms}))
    if missing_effects:
        hard_gaps.append(gap(patch_id, "effect_logs_missing", {"missing": missing_effects}))
    if not list_values(row.get("validation_commands")):
        hard_gaps.append(gap(patch_id, "validation_commands_missing", {}))
    if not str(row.get("rollback_notes") or "").strip():
        hard_gaps.append(gap(patch_id, "rollback_notes_missing", {}))
    if len(list_values(row.get("non_claims"))) < 2:
        warnings.append(gap(patch_id, "weak_non_claims", {"count": len(list_values(row.get("non_claims")))}, severity="warning"))
    return {
        "id": patch_id,
        "artifact": str(row.get("artifact") or ""),
        "artifact_sha256": file_hash(resolve(str(row.get("artifact") or ""))) if str(row.get("artifact") or "") else "",
        "affected_atoms": affected,
        "validation_command_count": len(list_values(row.get("validation_commands"))),
        "effect_log_count": len(effect_refs),
        "obligations": list_values(row.get("obligations")),
        "side_effects": list_values(row.get("side_effects")),
        "rollback_notes": str(row.get("rollback_notes") or ""),
        "non_claims": list_values(row.get("non_claims")),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def audit_fixture(row: dict[str, Any], atoms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fixture_id = str(row.get("id") or "<missing-id>")
    missing = sorted(REQUIRED_FIXTURE_FIELDS - set(row))
    broken_atom_id = str(row.get("broken_atom_id") or "")
    atom = atoms.get(broken_atom_id)
    repair_scope = set(str(x) for x in list_values(row.get("repair_scope")))
    atom_scope = set(str(x) for x in (atom or {}).get("repair_scope", []))
    inside_scope = bool(repair_scope and atom_scope and repair_scope.issubset(atom_scope))
    ledger = list_values(row.get("scope_change_ledger"))
    expected = str(row.get("expected_scope_result") or "")
    evidence_ok = all(evidence_anchor_present(str(item)) for item in list_values(row.get("current_repair_evidence")))
    hard_gaps = []
    warnings = []
    if missing:
        hard_gaps.append(gap(fixture_id, "missing_fixture_fields", {"fields": missing}))
    if atom is None:
        hard_gaps.append(gap(fixture_id, "broken_atom_missing", {"broken_atom_id": broken_atom_id}))
    if not inside_scope and not ledger:
        hard_gaps.append(gap(fixture_id, "scope_change_without_ledger", {"repair_scope": sorted(repair_scope), "atom_scope": sorted(atom_scope)}))
    if expected == "inside_declared_repair_scope" and not inside_scope:
        hard_gaps.append(gap(fixture_id, "expected_inside_scope_but_not_inside", {"repair_scope": sorted(repair_scope), "atom_scope": sorted(atom_scope)}))
    if not evidence_ok:
        hard_gaps.append(gap(fixture_id, "current_repair_evidence_missing", {"evidence": list_values(row.get("current_repair_evidence"))}))
    return {
        "id": fixture_id,
        "broken_atom_id": broken_atom_id,
        "inside_repair_scope": inside_scope,
        "scope_change_ledger_count": len(ledger),
        "current_repair_evidence_ok": evidence_ok,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def evidence_anchor_present(text: str) -> bool:
    if " contains " in text:
        path_text, anchor = text.split(" contains ", 1)
        path = resolve(path_text.strip())
        anchor = anchor.strip()
        return path.exists() and anchor in path.read_text(encoding="utf-8", errors="ignore")
    if " emits " in text:
        path_text, anchor = text.split(" emits ", 1)
        path = resolve(path_text.strip())
        anchor = anchor.strip()
        return path.exists() and anchor in path.read_text(encoding="utf-8", errors="ignore")
    if " reports " in text:
        path_text, anchor = text.split(" reports ", 1)
        path = resolve(path_text.strip())
        return path.exists() and anchor.strip() in path.read_text(encoding="utf-8", errors="ignore")
    return bool(text)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Semantic Patch Gate",
        "",
        f"- trigger_state: `{report['trigger_state']}`",
        f"- atoms: `{report['summary']['atom_count']}` anchored `{report['summary']['anchored_atom_count']}`",
        f"- patches: `{report['summary']['patch_count']}`",
        f"- effect logs: `{report['summary']['effect_log_count']}`",
        f"- fixtures inside scope: `{report['summary']['fixtures_inside_scope']}`",
        f"- hard gaps: `{report['summary']['hard_gap_count']}` warnings: `{report['summary']['warning_count']}`",
        "",
        "## Atoms",
        "",
    ]
    for atom in report["atoms"]:
        lines.append(f"- `{atom['id']}` type=`{atom['type']}` anchor_present=`{atom['source_anchor_present']}` artifact=`{atom['artifact']}`")
    lines.extend(["", "## Patches", ""])
    for patch in report["patches"]:
        lines.append(f"- `{patch['id']}` atoms=`{len(patch['affected_atoms'])}` validations=`{patch['validation_command_count']}` effects=`{patch['effect_log_count']}`")
    lines.extend(["", "## Hard Gaps", ""])
    if report["hard_gaps"]:
        for item in report["hard_gaps"]:
            lines.append(f"- `{item['id']}` `{item['kind']}`: `{json.dumps(item['evidence'], sort_keys=True)}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Rules", ""])
    for key, value in report["rules"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report["policy"],
        "created_utc": report["created_utc"],
        "trigger_state": report["trigger_state"],
        "summary": report["summary"],
        "hard_gaps": report["hard_gaps"],
        "warnings": report["warnings"],
    }


def materialized_atom(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ["id", "type", "artifact", "artifact_sha256", "obligation_ids", "repair_scope", "authority_boundary", "source_anchor_present"]}


def materialized_patch(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ["id", "artifact", "artifact_sha256", "affected_atoms", "obligations", "side_effects", "rollback_notes", "non_claims"]}


def materialized_effect(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ["id", "command", "input_count", "output_count", "allowed_effects", "forbidden_effects", "replay_status"]}


def gap(item_id: str, kind: str, evidence: dict[str, Any], severity: str = "hard") -> dict[str, Any]:
    return {"id": item_id, "kind": kind, "severity": severity, "evidence": evidence}


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"id": name, "kind": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_hash(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def list_values(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
