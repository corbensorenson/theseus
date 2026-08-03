#!/usr/bin/env python3
"""Exercise the canonical Semantic-IR owner through the real production path.

Consumed P4 evaluator-only oracles are reused solely as mechanics fixtures.
They are never rescored as model candidates and produce no claim, D1, D2,
training, serving, or book-support credit.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_p4_cognitive_compilation as p4
import theseus_p4s_cognitive_compilation as p4s
import theseus_semantic_ir_production as production
import theseus_semantic_ir_v2 as v2
import theseus_semantic_ir_v2r2 as v2r2


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "configs" / "theseus_p4v2r2r2_task_pool.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_conformance.json"
POLICY = "project_theseus_semantic_ir_production_conformance_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=p2a.rel(POOL))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = run_conformance(p2a.resolve(args.pool))
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "policy": report["policy"],
        "trigger_state": report["trigger_state"],
        "faults": report["faults"],
        "coverage": report["coverage"],
        "counters": report["counters"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def run_conformance(pool_path: Path = POOL) -> dict[str, Any]:
    started = time.perf_counter()
    pool = p2a.read_json(pool_path)
    faults: list[str] = []
    task_receipts: list[dict[str, Any]] = []
    operation_counts = {"REPLACE": 0, "INSERT_BEFORE": 0, "INSERT_AFTER": 0}
    corruption_counts: dict[str, int] = {
        "source_identity": 0,
        "role_identity": 0,
        "change_coverage": 0,
        "non_goal_mutation": 0,
        "target_hash": 0,
        "coordinate_identity": 0,
        "unparsed_text": 0,
    }

    for row in p2a.dicts(pool.get("tasks")):
        stem = str(row.get("stem") or "")
        task_path = ROOT / str(row.get("task") or "")
        oracle_path = ROOT / str(row.get("treatment_transport_oracle_ir") or "")
        if p2a.sha256_file(task_path) != str(row.get("task_sha256") or ""):
            faults.append(f"task_binding_invalid:{stem}")
            continue
        if p2a.sha256_file(oracle_path) != str(
            row.get("treatment_transport_oracle_ir_sha256") or ""
        ):
            faults.append(f"oracle_binding_invalid:{stem}")
            continue
        task = p2a.read_json(task_path)
        upgraded = upgrade_evaluator_oracle(
            oracle_path.read_text(encoding="utf-8"), task
        )
        with tempfile.TemporaryDirectory(prefix=f"theseus-semantic-production-{stem}-") as tmp:
            root = Path(tmp) / "source"
            p2a.extract_source_archive(
                p2a.resolve(str(task.get("source_archive") or "")),
                root,
                str(task.get("source_archive_root") or ""),
            )
            baseline = p2a.inventory(root)
            symbols = p4s.semantic_scope_symbol_table(root, task)
            common = render_common_context(root, task, symbols)
            prompt = production.render_prompt(task, common)
            parsed = production.parse(upgraded, task, root)
            verification = p4.verify_provisional(root, baseline, task, parsed)
            repair = production.render_repair_prompt(
                prompt, upgraded, p2a.strings(parsed.get("faults")), verification
            )
            visible_pass = p2a.mapping(
                verification.get("visible_verifier")
            ).get("passed") is True
            if parsed.get("faults") or not p2a.dicts(parsed.get("actions")):
                faults.append(f"parse_or_lower_red:{stem}")
            if verification.get("apply_faults") or not visible_pass:
                faults.append(f"apply_or_visible_red:{stem}")
            if upgraded not in repair:
                faults.append(f"repair_artifact_not_complete:{stem}")
            for unit in p2a.dicts(parsed.get("units")):
                operation = str(unit.get("operation") or "")
                if operation in operation_counts:
                    operation_counts[operation] += 1

            coordinate_variant = with_redundant_exact_coordinates(
                upgraded, symbols
            )
            coordinate_parsed = production.parse(coordinate_variant, task, root)
            if (
                coordinate_parsed.get("faults")
                or p2a.stable_hash(coordinate_parsed.get("actions"))
                != p2a.stable_hash(parsed.get("actions"))
            ):
                faults.append(f"exact_coordinate_round_trip_red:{stem}")

            corruptions = corruption_variants(upgraded, task, symbols)
            rejected: dict[str, bool] = {}
            for name, mutation in corruptions.items():
                result = production.parse(mutation, task, root)
                rejected[name] = bool(result.get("faults")) and not result.get("actions")
                if rejected[name]:
                    corruption_counts[name] += 1
                else:
                    faults.append(f"corruption_not_rejected:{stem}:{name}")

            task_receipts.append({
                "stem": stem,
                "task_sha256": p2a.sha256_file(task_path),
                "evaluator_only_oracle_sha256": p2a.sha256_file(oracle_path),
                "upgraded_oracle_sha256": p2a.sha256_text(upgraded),
                "prompt_sha256": p2a.sha256_text(prompt),
                "repair_prompt_sha256": p2a.sha256_text(repair),
                "parse_lower_actions": len(p2a.dicts(parsed.get("actions"))),
                "apply_faults": p2a.strings(verification.get("apply_faults")),
                "visible_verifier_passed": visible_pass,
                "exact_coordinate_round_trip": not coordinate_parsed.get("faults"),
                "corruption_rejections": rejected,
                "candidate_or_control_calls": 0,
                "hidden_evaluator_calls": 0,
            })

    synthetic_operations, synthetic_faults = synthetic_operation_mechanics()
    faults.extend(synthetic_faults)
    for operation, count in synthetic_operations.items():
        operation_counts[operation] += count

    trigger = "GREEN" if not faults else "RED"
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": trigger,
        "faults": sorted(set(faults)),
        "bindings": {
            "pool": {"path": p2a.rel(pool_path), "sha256": p2a.sha256_file(pool_path)},
            "production_owner": {
                "path": p2a.rel(Path(production.__file__)),
                "sha256": p2a.sha256_file(Path(production.__file__)),
            },
        },
        "coverage": {
            "consumed_evaluator_only_mechanics_fixtures": len(task_receipts),
            "parse_lower_apply_visible_passes": sum(
                not receipt["apply_faults"] and receipt["visible_verifier_passed"]
                and receipt["parse_lower_actions"] > 0
                for receipt in task_receipts
            ),
            "exact_coordinate_round_trips": sum(
                receipt["exact_coordinate_round_trip"] for receipt in task_receipts
            ),
            "operation_mechanics": operation_counts,
            "corruption_rejections": corruption_counts,
            "complete_first_artifact_in_repair_prompt": all(
                receipt["repair_prompt_sha256"] for receipt in task_receipts
            ),
        },
        "tasks": task_receipts,
        "counters": {
            "candidate_or_control_calls": 0,
            "hidden_evaluator_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
        "project_selected_quality_token_cap": None,
        "maximum_inference": (
            "The canonical production schema, role-aware parser, target resolver, "
            "deterministic lowerer, disposable applier, visible verifier, corruption "
            "interventions, and complete repair prompt conform on retained evaluator-only "
            "mechanics fixtures. This is non-claim mechanics evidence only and does not "
            "rescore a model candidate or establish competence, treatment effect, D1, "
            "D2, serving, training, or ASI Stack book support."
        ),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def upgrade_evaluator_oracle(oracle: str, task: dict[str, Any]) -> str:
    normalized, _ = v2r2.normalize_with_receipt(oracle)
    source = re.search(r"^SOURCE ([a-f0-9]{64})$", normalized, re.MULTILINE)
    loss = re.search(r"^LOSS (NONE|[A-Z0-9_,]+)$", normalized, re.MULTILINE)
    if source is None or loss is None:
        raise ValueError("historical_oracle_envelope_invalid")
    roles = production.obligation_roles(task)
    non_goals = set(roles["non_goal"])
    lines = [
        production.HEADER,
        f"SOURCE {source.group(1)}",
        f"ALL_OBLIGATIONS {production.encode_ids(roles['all'])}",
        f"CHANGE_OBLIGATIONS {production.encode_ids(roles['change'])}",
        f"PRESERVE_OBLIGATIONS {production.encode_ids(roles['preserve'])}",
        f"NON_GOAL_OBLIGATIONS {production.encode_ids(roles['non_goal'])}",
    ]
    for match in v2.UNIT_RE.finditer(normalized):
        unit_id, refs, operation, path, node_id, node_hash, replacement = match.groups()
        scoped = [value for value in refs.split(",") if value not in non_goals]
        lines.extend([
            f"UNIT {unit_id}",
            f"OBLIGATIONS {production.encode_ids(scoped)}",
            f"OP {operation}",
            f"PATH {path}",
            f"NODE {node_id}",
            f"NODE_SHA {node_hash}",
            "<<<",
            replacement,
            ">>>",
            "END_UNIT",
        ])
    lines.extend([f"LOSS {loss.group(1)}", "END"])
    return "\n".join(lines)


def with_redundant_exact_coordinates(
    artifact: str, symbols: dict[str, Any]
) -> str:
    node = re.search(r"^NODE ([A-Z0-9-]+)$", artifact, re.MULTILINE)
    path = re.search(r"^PATH ([^\n ]+)$", artifact, re.MULTILINE)
    if node is None or path is None:
        return artifact
    symbol = next(
        row for row in p2a.dicts(symbols.get("nodes")) if row.get("id") == node.group(1)
    )
    decorated = (
        f"{path.group(1)}:{symbol['start_line']}:{symbol['start_col']}-"
        f"{symbol['end_line']}:{symbol['end_col']}"
    )
    return artifact[: path.start(1)] + decorated + artifact[path.end(1) :]


def corruption_variants(
    artifact: str, task: dict[str, Any], symbols: dict[str, Any]
) -> dict[str, str]:
    roles = production.obligation_roles(task)
    change = roles["change"][0]
    non_goal = roles["non_goal"][0] if roles["non_goal"] else "O999"
    node_hash = re.search(r"^NODE_SHA ([a-f0-9]{64})$", artifact, re.MULTILINE)
    source = re.search(r"^SOURCE ([a-f0-9]{64})$", artifact, re.MULTILINE)
    unit_refs = re.search(r"^OBLIGATIONS ([A-Z0-9_,]+)$", artifact, re.MULTILINE)
    coordinate = with_redundant_exact_coordinates(artifact, symbols)
    coordinate_path = re.search(r"^PATH ([^\n ]+)$", coordinate, re.MULTILINE)
    assert node_hash and source and unit_refs and coordinate_path
    return {
        "source_identity": artifact[: source.start(1)] + "0" * 64 + artifact[source.end(1) :],
        "role_identity": artifact.replace(
            f"CHANGE_OBLIGATIONS {production.encode_ids(roles['change'])}",
            f"CHANGE_OBLIGATIONS {non_goal}",
            1,
        ),
        "change_coverage": remove_change_from_all_units(artifact, change),
        "non_goal_mutation": artifact[: unit_refs.start(1)]
        + unit_refs.group(1)
        + ","
        + non_goal
        + artifact[unit_refs.end(1) :],
        "target_hash": artifact[: node_hash.start(1)] + "0" * 64 + artifact[node_hash.end(1) :],
        "coordinate_identity": coordinate[: coordinate_path.end(1) - 1]
        + ("0" if coordinate[coordinate_path.end(1) - 1] != "0" else "1")
        + coordinate[coordinate_path.end(1) :],
        "unparsed_text": artifact.replace("LOSS NONE", "COMMENTARY\nLOSS NONE", 1),
    }


def remove_change_from_all_units(artifact: str, change_id: str) -> str:
    def replace(match: re.Match[str]) -> str:
        refs = [value for value in match.group(1).split(",") if value != change_id]
        return "OBLIGATIONS " + production.encode_ids(refs)

    return re.sub(
        r"^OBLIGATIONS ([A-Z0-9_,]+)$",
        replace,
        artifact,
        flags=re.MULTILINE,
    )


def synthetic_operation_mechanics() -> tuple[dict[str, int], list[str]]:
    counts = {"REPLACE": 0, "INSERT_BEFORE": 0, "INSERT_AFTER": 0}
    faults: list[str] = []
    for operation in counts:
        with tempfile.TemporaryDirectory(prefix=f"theseus-semantic-{operation.lower()}-") as tmp:
            root = Path(tmp)
            (root / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
            task = {
                "allowed_effect_paths": ["sample.py"],
                "candidate_visible_context": {
                    "reads": [{"path": "sample.py", "start_line": 1, "end_line": 1}]
                },
                "semantic_ir_contract": {"maximum_symbol_nodes": 20},
                "obligations": [
                    {"id": "O1", "kind": "require", "text": "exercise operation"}
                ],
                "obligation_dependencies": [],
                "visible_verifier": {
                    "command": ["python3", "-m", "py_compile", "sample.py"],
                    "timeout_seconds": 10,
                },
            }
            symbols = p4s.semantic_scope_symbol_table(root, task)
            target = next(row for row in symbols["nodes"] if row["node_type"] == "Assign")
            replacement = "VALUE = 2" if operation == "REPLACE" else "MARKER = 0"
            artifact = "\n".join([
                production.HEADER,
                f"SOURCE {symbols['source_digest']}",
                "ALL_OBLIGATIONS O1",
                "CHANGE_OBLIGATIONS O1",
                "PRESERVE_OBLIGATIONS NONE",
                "NON_GOAL_OBLIGATIONS NONE",
                "UNIT U1",
                "OBLIGATIONS O1",
                f"OP {operation}",
                "PATH sample.py",
                f"NODE {target['id']}",
                f"NODE_SHA {target['sha256']}",
                "<<<",
                replacement,
                ">>>",
                "END_UNIT",
                "LOSS NONE",
                "END",
            ])
            parsed = production.parse(artifact, task, root)
            apply_faults = p2a.apply_actions(root, p2a.dicts(parsed.get("actions")))
            verifier = p2a.run_visible_verifier(root, task) if not apply_faults else {}
            if (
                parsed.get("faults")
                or apply_faults
                or p2a.mapping(verifier).get("passed") is not True
            ):
                faults.append(f"synthetic_operation_red:{operation}")
            else:
                counts[operation] += 1
    return counts, faults


def render_common_context(
    root: Path, task: dict[str, Any], symbols: dict[str, Any]
) -> str:
    original = p4.semantic_symbol_table
    try:
        p4.semantic_symbol_table = lambda _root, _task: symbols
        return p4.render_common_context(root, task)
    finally:
        p4.semantic_symbol_table = original


if __name__ == "__main__":
    raise SystemExit(main())
