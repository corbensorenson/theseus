"""Convert private Decoder Plan IR rows into Code LM training rows.

Decoder Plan IR rows are the right shape for causal generation pressure, but
``code_lm_closure.py`` only ingests rows with a prompt, entry point, and
private solution body/expression. This adapter joins private Plan IR rows back
to their private source tasks and emits trainable Code LM rows with an enriched
decoder contract.

Public benchmark artifacts remain calibration-only. Public residual evidence
may influence high-level repair labels only when already present in the Plan IR
row; public prompts, tests, answers, and candidate code are never emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_PLAN_ROWS = Path("D:/ProjectTheseus/training_data/decoder_plan_ir/private_train/decoder_plan_ir_rows.jsonl")
DEFAULT_SOURCE_PATHS = [
    ROOT / "data/private_code_curriculum/code_lm_closure_private_pressure_private.jsonl",
    Path("D:/ProjectTheseus/training_data/high_transfer/private_train/type_and_return_shape_residual_code_lm_tasks.jsonl"),
    Path("D:/ProjectTheseus/training_data/high_transfer/private_train/admissibility_and_interface_residual_code_lm_tasks.jsonl"),
    Path("D:/ProjectTheseus/training_data/high_transfer/private_train/edge_conditions_residual_code_lm_tasks.jsonl"),
    Path("D:/ProjectTheseus/training_data/high_transfer/private_train/edge_contract_4card_residual_code_lm_tasks.jsonl"),
    Path("D:/ProjectTheseus/training_data/high_transfer/private_train/edge_contract_balanced_4card_private_curriculum_v2_residual_code_lm_tasks.jsonl"),
    Path("D:/ProjectTheseus/training_data/high_transfer/private_train/edge_case_full_body_private_curriculum_v1_residual_code_lm_tasks.jsonl"),
    Path("D:/ProjectTheseus/training_data/high_transfer/private_train/edge_contract_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl"),
    Path("D:/ProjectTheseus/training_data/high_transfer/private_train/algorithmic_planning_residual_code_lm_tasks.jsonl"),
    Path("D:/ProjectTheseus/training_data/high_transfer/private_train/execution_shaped_programs_residual_code_lm_tasks.jsonl"),
]
DEFAULT_ROWS_OUT = Path("D:/ProjectTheseus/training_data/decoder_plan_ir/private_train/decoder_plan_ir_code_lm_rows.jsonl")
DEFAULT_OUT = REPORTS / "decoder_plan_ir_code_lm_adapter.json"
DEFAULT_MARKDOWN = REPORTS / "decoder_plan_ir_code_lm_adapter.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-rows", default=str(DEFAULT_PLAN_ROWS))
    parser.add_argument(
        "--source-jsonl",
        action="append",
        default=[],
        help="Private Code LM source rows. May be supplied multiple times.",
    )
    parser.add_argument("--rows-out", default=str(DEFAULT_ROWS_OUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--max-rows", type=int, default=2400)
    parser.add_argument("--min-joined-rows", type=int, default=1000)
    args = parser.parse_args()

    started = time.perf_counter()
    plan_path = resolve(args.plan_rows)
    source_paths = [resolve(path) for path in args.source_jsonl] if args.source_jsonl else DEFAULT_SOURCE_PATHS
    source_rows = load_source_rows(source_paths)
    sources_by_task = {str(row.get("task_id") or ""): row for row in source_rows if row.get("task_id")}
    plan_rows = read_jsonl(plan_path)
    rows, skipped = adapt_rows(plan_rows, sources_by_task, max_rows=max(1, args.max_rows))
    rows_out = resolve(args.rows_out)
    write_jsonl(rows_out, rows)

    summary = summarize(rows, plan_rows, source_rows, skipped, started)
    gates = [
        gate("plan_rows_present", len(plan_rows) > 0, len(plan_rows)),
        gate("private_source_rows_present", len(source_rows) > 0, len(source_rows)),
        gate("joined_rows_above_floor", len(rows) >= max(1, args.min_joined_rows), {"rows": len(rows), "floor": args.min_joined_rows}),
        gate("all_rows_have_private_solution", all(has_private_solution(row) for row in rows), "solution_body or solution_expr present"),
        gate("all_rows_have_decoder_contract", all(isinstance(row.get("decoder_contract"), dict) and row["decoder_contract"] for row in rows), "contract required"),
        gate("no_public_benchmark_training_rows", all(not row.get("public_benchmark") for row in rows), "private-only"),
        gate("no_public_tests_or_solutions", public_leak_flag_count(rows) == 0, "public tests/solutions flags all false"),
        gate("plan_contract_rate_high", summary["contract_row_rate"] >= 0.98, summary["contract_row_rate"]),
    ]
    report = {
        "policy": "project_theseus_decoder_plan_ir_code_lm_adapter_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "RED",
        "passed": all(row["passed"] for row in gates),
        "purpose": "Make private Decoder Plan IR pressure trainable by Code LM closure without public benchmark leakage.",
        "inputs": {
            "plan_rows": display(plan_path),
            "source_jsonl": [display(path) for path in source_paths],
        },
        "outputs": {
            "rows_out": display(rows_out),
            "report": display(resolve(args.out)),
            "markdown": display(resolve(args.markdown_out)),
        },
        "summary": summary,
        "gates": gates,
        "sample_rows": rows[:5],
        "rules": {
            "training_surface": "private generated/local traces only",
            "public_benchmarks": "calibration-only; public prompts/tests/answers/candidate code are not emitted",
            "plan_ir_role": "causal skeleton/repair pressure inside decoder_contract, not promotion evidence",
        },
        "next_actions": [
            "include decoder_plan_ir_code_lm_rows.jsonl in the next private_pressure_private_closure",
            "require decoder_v2_private_ablation_gate before one public 4-card calibration",
            "if public transfer stays flat, pass exact aggregate residual cluster to teacher for one architecture experiment",
        ],
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    ingest_self(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0


def adapt_rows(
    plan_rows: list[dict[str, Any]],
    sources_by_task: dict[str, dict[str, Any]],
    *,
    max_rows: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    for plan in plan_rows:
        if len(rows) >= max_rows:
            break
        if bool(plan.get("public_benchmark")):
            skipped["public_plan_row"] += 1
            continue
        task_id = str(plan.get("task_id") or "")
        source = sources_by_task.get(task_id)
        if not source:
            skipped["source_missing"] += 1
            continue
        if bool(source.get("public_benchmark")):
            skipped["public_source_row"] += 1
            continue
        if not has_private_solution(source):
            skipped["source_missing_solution"] += 1
            continue
        prompt = str(source.get("prompt") or "").strip()
        entry_point = str(source.get("entry_point") or plan.get("entry_point") or "").strip()
        if not prompt or not entry_point:
            skipped["missing_prompt_or_entry_point"] += 1
            continue
        out_task_id = safe_name(f"decoder_plan_ir_{task_id}")
        if out_task_id in seen:
            skipped["duplicate"] += 1
            continue
        row = dict(source)
        row.update(
            {
                "task_id": out_task_id,
                "source_task_id": str(source.get("source_task_id") or task_id),
                "source_id": "local_generated_decoder_plan_ir_code_lm_adapter",
                "card_id": "private_decoder_plan_ir_code_lm",
                "split": "train",
                "dataset_id": "dataset.decoder_plan_ir_code_lm.private.v1",
                "benchmark_evidence_level": "private_plan_ir_generated_training_only",
                "public_benchmark": False,
                "public_benchmark_solutions_included": False,
                "public_tests_included": False,
                "license_spdx": str(source.get("license_spdx") or plan.get("license_spdx") or "local-generated-provenance-only"),
                "decoder_contract": merged_contract(source, plan),
                "provenance": merged_provenance(source, plan),
                "tags": merged_tags(source, plan),
            }
        )
        rows.append(row)
        seen.add(out_task_id)
    return rows, skipped


def merged_contract(source: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    source_contract = source.get("decoder_contract") if isinstance(source.get("decoder_contract"), dict) else {}
    generation_plan = source_contract.get("generation_plan") if isinstance(source_contract.get("generation_plan"), dict) else {}
    plan_contract = {
        "policy": "project_theseus_decoder_plan_ir_code_lm_contract_v1",
        "plan_ir_row_id": str(plan.get("row_id") or ""),
        "training_role": "causal_skeleton_choice_and_private_execution_repair",
        "causal_order": plan.get("causal_order") if isinstance(plan.get("causal_order"), list) else [
            "signature",
            "argument_roles",
            "return_contract",
            "semantic_family",
            "state_variables",
            "branch_loop_skeleton",
            "library_api_plan",
            "body",
            "private_execution_repair",
        ],
        "signature": plan.get("signature") if isinstance(plan.get("signature"), dict) else {},
        "argument_roles": plan.get("argument_roles") if isinstance(plan.get("argument_roles"), dict) else source_contract.get("argument_roles", {}),
        "return_contract": plan.get("return_contract") if isinstance(plan.get("return_contract"), dict) else source_contract.get("return_contract", {}),
        "return_shape": get_path(plan, ["return_contract", "shape"], source_contract.get("return_shape")),
        "type_family": str(plan.get("semantic_family") or source_contract.get("type_family") or ""),
        "semantic_family": str(plan.get("semantic_family") or source_contract.get("type_family") or ""),
        "state_variables": plan.get("state_variables") if isinstance(plan.get("state_variables"), list) else [],
        "branch_loop_skeleton": plan.get("branch_loop_skeleton") if isinstance(plan.get("branch_loop_skeleton"), list) else [],
        "library_api_plan": plan.get("library_api_plan") if isinstance(plan.get("library_api_plan"), list) else [],
        "edge_case_obligations": plan.get("edge_case_obligations") if isinstance(plan.get("edge_case_obligations"), list) else [],
        "repair_signals": plan.get("repair_signals") if isinstance(plan.get("repair_signals"), list) else [],
        "sts_conditioning_hints": plan.get("sts_conditioning_hints") if isinstance(plan.get("sts_conditioning_hints"), list) else [],
        "generation_plan": {
            **generation_plan,
            "policy": "signature -> argument_roles -> return_contract -> semantic_family -> state_variables -> branch_loop_skeleton -> body -> private_execution_repair",
            "skeleton_bias": plan.get("branch_loop_skeleton") if isinstance(plan.get("branch_loop_skeleton"), list) else [],
            "repair_strategy": "use Plan IR verifier feedback to choose skeleton and local state before body decoding",
            "verifier_feedback": plan.get("repair_signals") if isinstance(plan.get("repair_signals"), list) else [],
            "public_solutions_used": False,
            "public_tests_used": False,
        },
        "score_semantics": "private Plan IR Code LM pressure only; not benchmark evidence",
        "public_solutions_used": False,
        "public_tests_used": False,
    }
    merged = dict(source_contract)
    merged.update(plan_contract)
    return merged


def merged_provenance(source: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    provenance = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
    out = dict(provenance)
    out.update(
        {
            "policy": "project_theseus_decoder_plan_ir_code_lm_adapter_v1",
            "source_private_task_id": str(plan.get("task_id") or source.get("task_id") or ""),
            "plan_ir_row_id": str(plan.get("row_id") or ""),
            "plan_ir_source_task_hash": str(plan.get("source_task_hash") or ""),
            "public_benchmark_answers_used": False,
            "public_benchmark_solutions_used": False,
            "public_tests_used": False,
            "public_task_ids_hashed_only": True,
            "public_residual_categories_only": True,
        }
    )
    return out


def merged_tags(source: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    tags = []
    if isinstance(source.get("tags"), list):
        tags.extend(str(tag) for tag in source["tags"] if str(tag))
    tags.extend(
        [
            "decoder_plan_ir",
            "causal_skeleton_choice",
            "private_execution_repair",
            str(plan.get("semantic_family") or ""),
        ]
    )
    for item in plan.get("branch_loop_skeleton", []) if isinstance(plan.get("branch_loop_skeleton"), list) else []:
        tags.append(f"skeleton:{item}")
    for item in plan.get("repair_signals", []) if isinstance(plan.get("repair_signals"), list) else []:
        if str(item).startswith("public_pressure:"):
            tags.append(str(item))
        else:
            tags.append(f"repair:{item}")
    return sorted({safe_tag(tag) for tag in tags if safe_tag(tag)})


def summarize(
    rows: list[dict[str, Any]],
    plan_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    skipped: Counter[str],
    started: float,
) -> dict[str, Any]:
    contract_rows = sum(1 for row in rows if isinstance(row.get("decoder_contract"), dict) and row["decoder_contract"])
    body_rows = sum(1 for row in rows if has_private_solution(row))
    public_flags = public_leak_flag_count(rows)
    semantic_counts = Counter(str(get_path(row, ["decoder_contract", "semantic_family"], "unknown")) for row in rows)
    skeleton_counts = Counter(
        str(kind)
        for row in rows
        for kind in (get_path(row, ["decoder_contract", "branch_loop_skeleton"], []) or [])
    )
    repair_counts = Counter(
        str(kind)
        for row in rows
        for kind in (get_path(row, ["decoder_contract", "repair_signals"], []) or [])
    )
    return {
        "plan_ir_row_count": len(plan_rows),
        "source_row_count": len(source_rows),
        "code_lm_row_count": len(rows),
        "joined_row_count": len(rows),
        "skipped_counts": dict(skipped),
        "contract_row_count": contract_rows,
        "private_solution_row_count": body_rows,
        "public_leak_flag_count": public_flags,
        "contract_row_rate": round(contract_rows / max(1, len(rows)), 6),
        "private_solution_row_rate": round(body_rows / max(1, len(rows)), 6),
        "semantic_family_counts": dict(semantic_counts),
        "skeleton_kind_counts": dict(skeleton_counts),
        "repair_signal_counts": dict(repair_counts),
        "private_plan_ir_rows": len(rows),
        "rows_out": display(DEFAULT_ROWS_OUT),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def load_source_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            task_id = str(row.get("task_id") or "")
            if not task_id or task_id in seen:
                continue
            if bool(row.get("public_benchmark")):
                continue
            license_spdx = str(row.get("license_spdx") or "")
            if not license_spdx or license_spdx.lower() in {"unknown", "noassertion"}:
                continue
            rows.append(row)
            seen.add(task_id)
    return rows


def has_private_solution(row: dict[str, Any]) -> bool:
    return bool(str(row.get("solution_body") or "").strip() or str(row.get("solution_expr") or "").strip())


def public_leak_flag_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if bool(row.get("public_benchmark")):
            count += 1
        if bool(row.get("public_benchmark_solutions_included")):
            count += 1
        if bool(row.get("public_tests_included")):
            count += 1
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        if bool(provenance.get("public_benchmark_answers_used")) or bool(provenance.get("public_tests_used")):
            count += 1
        contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
        if bool(contract.get("public_solutions_used")) or bool(contract.get("public_tests_used")):
            count += 1
    return count


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return rows
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Decoder Plan IR Code LM Adapter",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Plan IR rows: `{summary.get('plan_ir_row_count')}`",
        f"- Code LM rows: `{summary.get('code_lm_row_count')}`",
        f"- Contract row rate: `{summary.get('contract_row_rate')}`",
        f"- Public leak flags: `{summary.get('public_leak_flag_count')}`",
        "",
        "## Gates",
        "",
    ]
    for gate_row in report.get("gates", []):
        lines.append(f"- `{gate_row.get('passed')}` {gate_row.get('name')}: {gate_row.get('evidence')}")
    lines.extend(["", "## Next Actions", ""])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def ingest_self(path: Path, payload: dict[str, Any]) -> None:
    try:
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        import report_evidence_store  # type: ignore

        report_evidence_store.ingest_report_path(report_evidence_store.DEFAULT_DB, path, payload=payload)
    except Exception:
        return


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value).strip())
    return cleaned.strip("_")[:180] or "private_task"


def safe_tag(value: str) -> str:
    cleaned = str(value).strip().replace(" ", "_")
    return cleaned[:120]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def short_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
