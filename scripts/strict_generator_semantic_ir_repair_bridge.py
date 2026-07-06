#!/usr/bin/env python3
"""Lower strict-generator decode failures into semantic-IR repair obligations.

This bridge turns real strict-generator failure evidence into localized
semantic atoms and patch obligations. It is deliberately not a decoder, repair
renderer, training-row writer, or promotion gate. It reads only generated
candidate/prefix diagnostics and aggregate verifier labels already present in a
strict decode report, then names the next repair surface:

failed prefix state -> semantic atom -> localized repair patch -> validation.

No public benchmark payloads, tests, solutions, answer templates, teacher
output, deterministic tools, fallback returns, or hidden target fields are
used or emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECODE_REPORT = ROOT / "reports" / "strict_generator_mlx_decode_eval_closed_state_transition_smoke_broad4_20260706.json"
DEFAULT_OUT = ROOT / "reports" / "strict_generator_semantic_ir_repair_bridge.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "strict_generator_semantic_ir_repair_bridge.md"
DEFAULT_ATOMS = ROOT / "reports" / "strict_generator_semantic_ir_repair_atoms.jsonl"
DEFAULT_PATCHES = ROOT / "reports" / "strict_generator_semantic_ir_repair_patches.jsonl"


NO_CHEAT = {
    "public_training_rows_written": 0,
    "external_inference_calls": 0,
    "fallback_return_count": 0,
    "candidate_generation_credit": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-report", default=rel(DEFAULT_DECODE_REPORT))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--atoms-out", default=rel(DEFAULT_ATOMS))
    parser.add_argument("--patches-out", default=rel(DEFAULT_PATCHES))
    parser.add_argument("--max-tasks", type=int, default=32)
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.atoms_out), report.get("semantic_atoms", []))
    write_jsonl(resolve(args.patches_out), report.get("semantic_repair_patches", []))
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    decode_path = resolve(args.decode_report)
    decode = read_json(decode_path)
    if not decode:
        report = base_report(started, decode_path)
        report.update(
            {
                "trigger_state": "YELLOW",
                "summary": {
                    **report["summary"],
                    "decode_report_present": False,
                    "semantic_atom_count": 0,
                    "repair_patch_count": 0,
                    "warning_count": 1,
                    "hard_gap_count": 0,
                },
                "warnings": [gap("decode_report_missing", {"path": rel(decode_path)}, severity="warning")],
                "gates": [gate("decode_report_present", False, "soft", rel(decode_path))],
            }
        )
        return report

    starvation_rows = strict_decode_starvation_rows(decode, max_tasks=max(1, int(args.max_tasks or 1)))
    atoms, patches = build_atoms_and_patches(decode_path, starvation_rows)
    forbidden = forbidden_boundary_counts(decode)
    hard_gates = [
        gate("decode_report_present", True, "hard", rel(decode_path)),
        gate("strict_decode_report_policy", decode.get("policy") == "project_theseus_strict_generator_mlx_decode_eval_v1", "hard", decode.get("policy")),
        gate("no_public_training_rows", forbidden["public_training_rows"] == 0, "hard", forbidden["public_training_rows"]),
        gate("no_runtime_external_inference", forbidden["external_inference_calls"] == 0, "hard", forbidden["external_inference_calls"]),
        gate("no_fallback_credit", forbidden["fallback_credit_count"] == 0, "hard", forbidden["fallback_credit_count"]),
        gate("semantic_atoms_emitted_for_failures", len(atoms) > 0 if starvation_rows else True, "hard", len(atoms)),
        gate("localized_repair_patches_emitted", len(patches) > 0 if starvation_rows else True, "hard", len(patches)),
        gate("raw_task_or_category_labels_not_emitted", raw_task_label_leak_count(atoms + patches) == 0, "hard", raw_task_label_leak_count(atoms + patches)),
    ]
    hard_failed = [row for row in hard_gates if row["severity"] == "hard" and not row["passed"]]
    issue_counts = Counter(
        issue
        for row in atoms
        for issue in list(row.get("issue_labels") or [])
    )
    trigger_state = "RED" if hard_failed else "GREEN"
    report = base_report(started, decode_path)
    report.update(
        {
            "trigger_state": trigger_state,
            "decode_report": {
                "path": rel(decode_path),
                "sha256": file_hash(decode_path),
                "trigger_state": decode.get("trigger_state"),
                "policy": decode.get("policy"),
                "summary_hash": stable_hash(decode.get("summary")),
            },
            "summary": {
                **report["summary"],
                "decode_report_present": True,
                "decode_trigger_state": decode.get("trigger_state"),
                "split_count": len(dict_or_empty(dict_or_empty(decode.get("summary")).get("split_decode_starvation"))),
                "failure_task_count": len(starvation_rows),
                "semantic_atom_count": len(atoms),
                "repair_patch_count": len(patches),
                "issue_label_counts": dict(sorted(issue_counts.items())),
                "hard_gap_count": len(hard_failed),
                "warning_count": 0,
                "runtime_ms": int((time.perf_counter() - started) * 1000),
            },
            "gates": hard_gates,
            "hard_gaps": hard_failed,
            "semantic_atoms": atoms,
            "semantic_repair_patches": patches,
            "warnings": [],
        }
    )
    return report


def base_report(started: float, decode_path: Path) -> dict[str, Any]:
    return {
        "policy": "project_theseus_strict_generator_semantic_ir_repair_bridge_v1",
        "created_utc": now(),
        "purpose": "Bind strict-generator failures to semantic-IR localized repair obligations.",
        "decode_report": {"path": rel(decode_path), "present": decode_path.exists()},
        "summary": {
            "decode_report_present": decode_path.exists(),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "rules": {
            "input_boundary": "generated strict decode diagnostics and aggregate verifier labels only",
            "output_boundary": "semantic atoms and repair patches only; no training rows or candidate bodies",
            "no_credit": "bridge grants zero learned-generation, fallback, router, tool, or renderer credit",
            "raw_id_boundary": "raw task ids, category labels, and benchmark-family labels are hashed or omitted",
        },
        "non_claims": [
            "This bridge does not decode code.",
            "This bridge does not repair a candidate body.",
            "This bridge does not train a model.",
            "This bridge does not prove private or public behavior.",
        ],
        **NO_CHEAT,
    }


def strict_decode_starvation_rows(decode: dict[str, Any], *, max_tasks: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_starvation = dict_or_empty(dict_or_empty(decode.get("summary")).get("split_decode_starvation"))
    for split_name, split_payload in split_starvation.items():
        payload = dict_or_empty(split_payload)
        for task in list(payload.get("examples") or []):
            if len(rows) >= max_tasks:
                return rows
            task_dict = dict_or_empty(task)
            examples = [dict_or_empty(item) for item in list(task_dict.get("top_beam_examples") or [])]
            rows.append(
                {
                    "split": str(split_name),
                    "task_ref_hash": stable_hash(str(task_dict.get("task_id") or "")),
                    "top_beam_state_counts": dict_or_empty(task_dict.get("top_beam_state_counts")),
                    "completed_beam_count": int(task_dict.get("completed_beam_count") or 0),
                    "beam_count": int(task_dict.get("beam_count") or 0),
                    "stopped": bool(task_dict.get("stopped")),
                    "beam_examples": examples[:3],
                }
            )
    return rows


def build_atoms_and_patches(decode_path: Path, starvation_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    atoms: list[dict[str, Any]] = []
    patches: list[dict[str, Any]] = []
    for index, row in enumerate(starvation_rows):
        task_hash = str(row.get("task_ref_hash") or stable_hash(index))
        issue_labels = issue_labels_for_row(row)
        if not issue_labels:
            issue_labels = ["unclassified_decode_starvation"]
        atom_ids: list[str] = []
        for issue in issue_labels:
            atom_id = f"atom.strict_generator.{task_hash[:12]}.{issue}"
            atom_ids.append(atom_id)
            atoms.append(
                {
                    "id": atom_id,
                    "record_type": "semantic_ir_atom",
                    "type": "strict_generator_failure_atom",
                    "artifact": rel(decode_path),
                    "task_ref_hash": task_hash,
                    "issue_labels": [issue],
                    "source_anchor": "split_decode_starvation",
                    "obligation_ids": obligations_for_issue(issue),
                    "repair_scope": repair_scope_for_issue(issue),
                    "authority_boundary": "private_failure_analysis_only_no_generation_credit",
                    "evidence_refs": [rel(decode_path)],
                    "beam_state_counts": dict_or_empty(row.get("top_beam_state_counts")),
                    "prefix_observations": prefix_observations(row),
                    "support_state": "negative_behavior_evidence",
                    "score_semantics": "Semantic atom over generated private strict-decode failure evidence; not a training row or candidate.",
                    "uses_eval_tests_or_solutions": False,
                    "uses_public_data": False,
                    "uses_answer_metadata": False,
                    **NO_CHEAT,
                    "non_claims": [
                        "failure atom only",
                        "not learned-generation evidence",
                        "not a repaired candidate",
                    ],
                }
            )
        patch_id = f"patch.strict_generator.{task_hash[:12]}.localized_body_construction"
        patches.append(
            {
                "id": patch_id,
                "record_type": "semantic_ir_patch_obligation",
                "artifact": rel(decode_path),
                "affected_atoms": atom_ids,
                "obligations": sorted({ob for issue in issue_labels for ob in obligations_for_issue(issue)}),
                "repair_strategy": "semantic_ir_localized_body_construction",
                "repair_scope": sorted({scope for issue in issue_labels for scope in repair_scope_for_issue(issue)}),
                "validation_commands": [
                    "python3 scripts/strict_generator_semantic_ir_repair_bridge.py",
                    "python3 scripts/semantic_ir_obligation_gate.py",
                    "python3 scripts/strict_generator_mlx_decode_eval.py --checkpoint-report <new_checkpoint_report> --split broad_private_heldout --max-broad-rows 4 --execute",
                ],
                "scope_change_ledger_required": True,
                "expected_next_model_change": "trainable AST/state-transition head or semantic-IR localized repair path; no hidden renderer or fallback return",
                "rollback_notes": "Remove this repair obligation from Phase 13 evidence if future strict decode reports show the failure labels were misclassified; retain the source RED report as negative evidence.",
                "evidence_refs": [rel(decode_path)],
                "support_state": "repair_obligation_open",
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "uses_answer_metadata": False,
                **NO_CHEAT,
                "non_claims": [
                    "patch obligation only",
                    "not an applied code repair",
                    "not a model-quality claim",
                ],
            }
        )
    return atoms, patches


def issue_labels_for_row(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    state_counts = dict_or_empty(row.get("top_beam_state_counts"))
    if int(state_counts.get("missing_local_return") or 0) > 0:
        labels.append("missing_local_return_closure")
    if int(state_counts.get("current_line_starts_return") or 0) > 0:
        labels.append("unfinished_return_expression")
    observations = prefix_observations(row)
    if int(observations.get("max_repeated_guard_depth") or 0) >= 3:
        labels.append("repeated_nested_guard_without_progress")
    if int(observations.get("body_preview_with_dedent_count") or 0) > 0 and int(state_counts.get("missing_local_return") or 0) > 0:
        labels.append("block_exit_without_finalizer")
    return sorted(set(labels))


def prefix_observations(row: dict[str, Any]) -> dict[str, Any]:
    repeated_depth = 0
    dedent_count = 0
    return_prefix_count = 0
    preview_hashes: list[str] = []
    for beam in list(row.get("beam_examples") or []):
        body = str(dict_or_empty(beam).get("body_preview") or "")
        preview_hashes.append(stable_hash(body)[:16])
        repeated_depth = max(repeated_depth, repeated_guard_depth(body))
        dedent_count += body.count("\n") if "DEDENT:" in " ".join(str(tok) for tok in list(dict_or_empty(beam).get("decoded_token_tail") or [])) else 0
        tail_values = token_values(list(dict_or_empty(beam).get("decoded_token_tail") or []))
        if tail_values and tail_values[-1] == "return":
            return_prefix_count += 1
    return {
        "beam_example_count": len(list(row.get("beam_examples") or [])),
        "beam_preview_hashes": preview_hashes,
        "max_repeated_guard_depth": repeated_depth,
        "body_preview_with_dedent_count": dedent_count,
        "return_prefix_tail_count": return_prefix_count,
    }


def repeated_guard_depth(body: str) -> int:
    guards: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("if "):
            guards.append(re.sub(r"\s+", " ", stripped))
    if not guards:
        return 0
    counts = Counter(guards)
    return max(counts.values())


def token_values(tokens: list[Any]) -> list[str]:
    out = []
    for token in tokens:
        text = str(token)
        if ":" in text:
            out.append(text.split(":", 1)[1])
        else:
            out.append(text)
    return out


def obligations_for_issue(issue: str) -> list[str]:
    table = {
        "missing_local_return_closure": [
            "emit_final_top_level_return_after_blocks",
            "return_depends_on_visible_input_or_local_state",
        ],
        "unfinished_return_expression": [
            "complete_return_expression_before_eos",
            "validate_return_expression_static_value",
        ],
        "repeated_nested_guard_without_progress": [
            "bound_equivalent_guard_depth",
            "prefer_update_or_block_exit_after_guard",
        ],
        "block_exit_without_finalizer": [
            "close_control_block_then_emit_finalizer",
            "preserve_local_state_across_dedent",
        ],
        "unclassified_decode_starvation": [
            "classify_decode_starvation_before_training_claim",
        ],
    }
    return table.get(issue, ["classify_decode_starvation_before_training_claim"])


def repair_scope_for_issue(issue: str) -> list[str]:
    base = [
        "scripts/strict_generator_mlx_decode_eval.py",
        "scripts/neural_seed_token_decoder_support.py",
        "scripts/strict_generator_mlx_private_adaptation.py",
        "scripts/strict_generator_mlx_adaptation_weights.py",
    ]
    if issue in {"missing_local_return_closure", "block_exit_without_finalizer", "repeated_nested_guard_without_progress"}:
        base.append("scripts/neural_seed_token_model_backend.py")
    if issue == "unfinished_return_expression":
        base.append("scripts/neural_seed_expression_value_guard.py")
    return sorted(set(base))


def forbidden_boundary_counts(decode: dict[str, Any]) -> dict[str, int]:
    summary = dict_or_empty(decode.get("summary"))
    return {
        "public_training_rows": int(summary.get("public_training_rows") or decode.get("public_training_rows") or 0),
        "external_inference_calls": int(summary.get("external_inference_calls") or decode.get("external_inference_calls") or 0),
        "fallback_credit_count": int(summary.get("fallback_template_router_tool_credit_count") or 0),
    }


def raw_task_label_leak_count(rows: list[dict[str, Any]]) -> int:
    forbidden_keys = {"task_id", "source_task_id", "category", "family", "benchmark_card", "card_id"}
    count = 0
    for row in rows:
        for key in forbidden_keys:
            if key in row:
                count += 1
    return count


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def gap(kind: str, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"kind": kind, "severity": severity, "evidence": evidence}


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    summary = dict_or_empty(report.get("summary"))
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "summary": summary,
        "hard_gaps": report.get("hard_gaps", []),
        "warnings": report.get("warnings", []),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Strict Generator Semantic IR Repair Bridge",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Decode report present: `{summary.get('decode_report_present')}`",
        f"- Failure tasks: `{summary.get('failure_task_count', 0)}`",
        f"- Semantic atoms: `{summary.get('semantic_atom_count', 0)}`",
        f"- Repair patches: `{summary.get('repair_patch_count', 0)}`",
        f"- Issue labels: `{summary.get('issue_label_counts', {})}`",
        "",
        "## Non-Claims",
    ]
    for item in report.get("non_claims", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def stable_hash(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    else:
        text = str(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


if __name__ == "__main__":
    raise SystemExit(main())
