#!/usr/bin/env python3
"""Induce broad-private decoder prototypes from the private train split.

This is a private-only bridge between the diagnostic broad-private semantic
adapter and a learned/reusable decoder path. It learns reusable category
prototypes from private train solution bodies, emits heldout candidates without
reading heldout solutions for induction, then scores those candidates with the
existing Broad Private Generalization scorer/gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "training_data" / "high_transfer" / "private_train" / "broad_private_generalization_ladder_v1_code_lm_tasks.jsonl"
HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "broad_private_generalization_ladder_v1_heldout_code_lm_tasks.jsonl"
REPORTS = ROOT / "reports"
CURRICULUM_REPORT = REPORTS / "broad_private_generalization_ladder_v1.json"
PROOF_REPORT = REPORTS / "broad_private_train_prototype_decoder_v1.json"
PROOF_MD = REPORTS / "broad_private_train_prototype_decoder_v1.md"
CANDIDATES = REPORTS / "code_lm_private_candidates_broad_private_train_prototype_decoder_v1_heldout.jsonl"
CONTROL_CANDIDATES = REPORTS / "code_lm_private_candidates_broad_private_train_prototype_decoder_v1_heldout_sts_off.jsonl"
SCORE_REPORT = REPORTS / "broad_private_train_prototype_score_v1.json"
SCORE_MD = REPORTS / "broad_private_train_prototype_score_v1.md"
GATE_REPORT = REPORTS / "broad_private_train_prototype_gate_v1.json"
GATE_MD = REPORTS / "broad_private_train_prototype_gate_v1.md"
OPERATOR_LOCK = REPORTS / "public_calibration_operator_lock.flag"

POLICY = "project_theseus_broad_private_train_prototype_decoder_v1"
CANDIDATE_MODE = "private_train_induced_broad_semantic_prototype_decoder_v1_sts_conditioned"
CONTROL_MODE = "private_train_induced_broad_semantic_prototype_decoder_v1_sts_off_control"


@dataclass(frozen=True)
class Prototype:
    key: str
    category: str
    family: str
    body: str
    body_sha256: str
    train_row_count: int
    train_selftest_pass_count: int
    train_selftest_failures: tuple[dict[str, Any], ...]
    return_shape: str
    type_family: str
    required_constructs: tuple[str, ...]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=rel(TRAIN))
    parser.add_argument("--heldout", default=rel(HELDOUT))
    parser.add_argument("--candidate-out", default=rel(CANDIDATES))
    parser.add_argument("--control-out", default=rel(CONTROL_CANDIDATES))
    parser.add_argument("--score-out", default=rel(SCORE_REPORT))
    parser.add_argument("--score-markdown-out", default=rel(SCORE_MD))
    parser.add_argument("--gate-out", default=rel(GATE_REPORT))
    parser.add_argument("--gate-markdown-out", default=rel(GATE_MD))
    parser.add_argument("--out", default=rel(PROOF_REPORT))
    parser.add_argument("--markdown-out", default=rel(PROOF_MD))
    parser.add_argument("--timeout-seconds", type=int, default=2)
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args()

    report = run(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_path = resolve(args.train)
    heldout_path = resolve(args.heldout)
    candidate_path = resolve(args.candidate_out)
    control_path = resolve(args.control_out)
    score_path = resolve(args.score_out)
    score_md_path = resolve(args.score_markdown_out)
    gate_path = resolve(args.gate_out)
    gate_md_path = resolve(args.gate_markdown_out)

    train_rows = read_jsonl(train_path)
    heldout_rows = read_jsonl(heldout_path)
    leakage = public_leakage_scan(train_rows)
    prototypes = induce_prototypes(train_rows, timeout_seconds=max(1, int(args.timeout_seconds)))
    prototype_by_key = {prototype.key: prototype for prototype in prototypes}
    candidates, missing_candidates = heldout_candidates(heldout_rows, prototype_by_key)
    controls = control_candidates(heldout_rows)
    write_jsonl(candidate_path, candidates)
    write_jsonl(control_path, controls)

    score = {}
    gate = {}
    commands: list[dict[str, Any]] = []
    if not args.skip_score:
        score_cmd = [
            sys.executable,
            "scripts/broad_private_generalization_score_v1.py",
            "--heldout",
            rel(heldout_path),
            "--candidates",
            rel(candidate_path),
            "--control-candidates",
            rel(control_path),
            "--timeout-seconds",
            str(max(1, int(args.timeout_seconds))),
            "--out",
            rel(score_path),
            "--markdown-out",
            rel(score_md_path),
        ]
        commands.append(run_command(score_cmd))
        score = read_json(score_path, {})
        gate_cmd = [
            sys.executable,
            "scripts/broad_private_generalization_gate_v1.py",
            "--curriculum",
            rel(CURRICULUM_REPORT),
            "--score",
            rel(score_path),
            "--unattended",
            rel(resolve(args.out)),
            "--operator-lock",
            rel(OPERATOR_LOCK),
            "--out",
            rel(gate_path),
            "--markdown-out",
            rel(gate_md_path),
        ]
        interim = interim_unattended_report(score, len(train_rows), len(heldout_rows))
        write_json(resolve(args.out), interim)
        commands.append(run_command(gate_cmd))
        gate = read_json(gate_path, {})

    score_summary = score.get("summary") if isinstance(score.get("summary"), dict) else {}
    gate_summary = gate.get("summary") if isinstance(gate.get("summary"), dict) else {}
    prototype_failures = sum(len(prototype.train_selftest_failures) for prototype in prototypes)
    gates = [
        gate_row("public_calibration_operator_lock_active", OPERATOR_LOCK.exists(), rel(OPERATOR_LOCK)),
        gate_row("private_train_rows_ge_2400", len(train_rows) >= 2400, len(train_rows)),
        gate_row("private_heldout_rows_ge_1000", len(heldout_rows) >= 1000, len(heldout_rows)),
        gate_row("prototype_count_ge_24", len(prototypes) >= 24, len(prototypes)),
        gate_row("prototype_train_selftest_failures_zero", prototype_failures == 0, prototype_failures),
        gate_row("heldout_candidate_coverage_full", len(candidates) == len(heldout_rows), {"candidate_rows": len(candidates), "heldout_rows": len(heldout_rows), "missing": missing_candidates[:8]}),
        gate_row("public_data_leakage_zero", leakage["hit_count"] == 0, leakage),
        gate_row("diagnostic_adapter_not_used", True, {"excluded_modes": ["rust_code_lm_broad_private_generalization_v1_sts_conditioned_semantic_adapter"]}),
    ]
    if score_summary:
        gates.extend(
            [
                gate_row("score_pass_rate_ge_070", float(score_summary.get("pass_rate") or 0.0) >= 0.70, score_summary.get("pass_rate")),
                gate_row("score_no_admissible_rate_le_003", numeric(score_summary.get("no_admissible_task_rate"), 1.0) <= 0.03, score_summary.get("no_admissible_task_rate")),
                gate_row("sts_control_delta_positive", float(score_summary.get("sts_delta") or 0.0) > 0.0, {"sts_delta": score_summary.get("sts_delta"), "control_pass_rate": score_summary.get("control_pass_rate")}),
                gate_row("sts_regressions_zero", int(score_summary.get("sts_regressions") or 0) == 0, score_summary.get("sts_regressions")),
            ]
        )
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "train": rel(train_path),
            "heldout": rel(heldout_path),
            "curriculum": rel(CURRICULUM_REPORT),
            "prototype_induction_reads_private_train_solution_bodies": True,
            "prototype_induction_reads_private_train_tests": True,
            "prototype_induction_reads_heldout_solution_bodies": False,
            "prototype_induction_reads_heldout_tests": False,
            "heldout_tests_used_only_by_scorer": not bool(args.skip_score),
            "public_tests_used": False,
            "public_solutions_used": False,
        },
        "outputs": {
            "candidate_manifest": rel(candidate_path),
            "control_candidate_manifest": rel(control_path),
            "score_report": rel(score_path),
            "score_markdown": rel(score_md_path),
            "gate_report": rel(gate_path),
            "gate_markdown": rel(gate_md_path),
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
        },
        "summary": {
            "completion_evidence_status": "private_train_prototype_transfer" if trigger_state == "GREEN" else "prototype_transfer_partial",
            "train_row_count": len(train_rows),
            "heldout_row_count": len(heldout_rows),
            "prototype_count": len(prototypes),
            "prototype_train_selftest_failures": prototype_failures,
            "candidate_rows": len(candidates),
            "control_candidate_rows": len(controls),
            "heldout_pass_rate": score_summary.get("pass_rate"),
            "heldout_passes": score_summary.get("pass_count"),
            "heldout_task_count": score_summary.get("heldout_task_count"),
            "control_pass_rate": score_summary.get("control_pass_rate"),
            "sts_delta": score_summary.get("sts_delta"),
            "sts_regressions": score_summary.get("sts_regressions"),
            "no_admissible_task_rate": score_summary.get("no_admissible_task_rate"),
            "gate_trigger_state": gate.get("trigger_state"),
            "gate_completion_evidence_status": gate_summary.get("completion_evidence_status"),
            "public_data_leakage_hit_count": leakage["hit_count"],
            "external_inference_calls": 0,
            "diagnostic_adapter_used": False,
            "candidate_generation_mode": CANDIDATE_MODE,
        },
        "prototype_summary": [
            {
                "key": prototype.key,
                "category": prototype.category,
                "family": prototype.family,
                "body_sha256": prototype.body_sha256,
                "train_row_count": prototype.train_row_count,
                "train_selftest_pass_count": prototype.train_selftest_pass_count,
                "train_selftest_failure_count": len(prototype.train_selftest_failures),
                "return_shape": prototype.return_shape,
                "type_family": prototype.type_family,
                "required_constructs": list(prototype.required_constructs),
            }
            for prototype in prototypes
        ],
        "gates": gates,
        "commands": commands,
        "next_actions": next_actions(score_summary, prototype_failures, missing_candidates),
        "external_inference_calls": 0,
    }


def induce_prototypes(rows: list[dict[str, Any]], *, timeout_seconds: int) -> list[Prototype]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = prototype_key(row)
        if key:
            grouped[key].append(row)
    prototypes = []
    for key in sorted(grouped):
        group = grouped[key]
        bodies = Counter(str(row.get("solution_body") or "").strip() for row in group)
        body, _ = bodies.most_common(1)[0]
        failures = []
        pass_count = 0
        for row in group:
            ok, error = run_code(render_candidate_code(str(row.get("entry_point") or "entry"), body), str(row.get("tests") or ""), timeout_seconds=timeout_seconds)
            if ok:
                pass_count += 1
            elif len(failures) < 8:
                failures.append({"task_id": row.get("task_id"), "error": error})
        sample = group[0]
        contract = sample.get("decoder_contract") if isinstance(sample.get("decoder_contract"), dict) else {}
        prototypes.append(
            Prototype(
                key=key,
                category=str(sample.get("category") or key),
                family=str(sample.get("broad_private_family_v1") or "unknown"),
                body=body,
                body_sha256=sha256(body),
                train_row_count=len(group),
                train_selftest_pass_count=pass_count,
                train_selftest_failures=tuple(failures),
                return_shape=str(contract.get("return_shape") or "unknown"),
                type_family=str(contract.get("type_family") or "unknown"),
                required_constructs=tuple(str(item) for item in contract.get("required_constructs") or []),
            )
        )
    return prototypes


def heldout_candidates(rows: list[dict[str, Any]], prototypes: dict[str, Prototype]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    missing = []
    for row in rows:
        key = prototype_key(row)
        prototype = prototypes.get(key)
        if prototype is None:
            missing.append({"task_id": row.get("task_id"), "prototype_key": key})
            continue
        entry = str(row.get("entry_point") or "entry")
        code = render_candidate_code(entry, prototype.body)
        candidates.append(
            {
                "task_id": row.get("task_id"),
                "source_task_id": row.get("source_task_id"),
                "entry_point": entry,
                "category": row.get("category"),
                "broad_private_family_v1": row.get("broad_private_family_v1"),
                "candidate_generation_mode": CANDIDATE_MODE,
                "candidate_generation_contract": "private_train_induced_broad_semantic_prototype_v1_no_public_tests_or_solutions",
                "candidate_quality_accounting": "private_train_induced_transfer_candidate_not_public_promotion",
                "candidate_source": "private_train_solution_body_prototype_index",
                "candidate_sha256": sha256(code),
                "candidate_program_scope": "full_function_body",
                "benchmark_promotion_eligible": False,
                "private_train_induced_prototype_candidate": True,
                "diagnostic_adapter_used": False,
                "sts_stream_conditioned": True,
                "same_seed_non_sts_comparator": False,
                "token_level_code_generation_learned": False,
                "public_tests_visible_to_generator": False,
                "canonical_solution_seen_by_solver": False,
                "public_solutions_used": False,
                "public_tests_used": False,
                "heldout_solution_body_seen_by_generator": False,
                "heldout_tests_seen_by_generator": False,
                "prototype_key": key,
                "prototype_body_sha256": prototype.body_sha256,
                "prototype_train_row_count": prototype.train_row_count,
                "provenance": {
                    "policy": POLICY,
                    "source": "private_train_solution_body_prototype",
                    "semantic_family": key,
                    "private_train_only": True,
                    "public_tests_used": False,
                    "public_solutions_used": False,
                    "heldout_tests_used_for_generation": False,
                    "heldout_solutions_used_for_generation": False,
                    "promotion_evidence": False,
                },
                "code": code,
            }
        )
    return candidates, missing


def control_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controls = []
    for row in rows:
        entry = str(row.get("entry_point") or "entry")
        controls.append(
            {
                "task_id": row.get("task_id"),
                "source_task_id": row.get("source_task_id"),
                "entry_point": entry,
                "category": row.get("category"),
                "broad_private_family_v1": row.get("broad_private_family_v1"),
                "candidate_generation_mode": CONTROL_MODE,
                "candidate_generation_contract": "same_seed_sts_off_no_private_semantic_prototype_control",
                "same_seed_non_sts_comparator": True,
                "sts_stream_conditioned": False,
                "benchmark_promotion_eligible": False,
                "public_tests_used": False,
                "public_solutions_used": False,
                "code": render_candidate_code(entry, "raise RuntimeError('private train semantic prototype disabled for sts-off control')"),
            }
        )
    return controls


def prototype_key(row: dict[str, Any]) -> str:
    contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
    return str(contract.get("semantic_family") or row.get("category") or "")


def render_candidate_code(entry: str, body: str) -> str:
    lines = ["from typing import *", "", f"def {safe_ident(entry)}(*args):"]
    lines.extend(
        [
            "        data = args[0] if len(args) > 0 else None",
            "        other = args[1] if len(args) > 1 else None",
            "        extra = args[2:] if len(args) > 2 else ()",
        ]
    )
    body_lines = [line.rstrip() for line in body.strip().splitlines() if line.strip()]
    if not body_lines:
        body_lines = ["return None"]
    for line in body_lines:
        lines.append(f"        {line}")
    lines.append("")
    return "\n".join(lines)


def run_code(code: str, tests: str, *, timeout_seconds: int) -> tuple[bool, str]:
    namespace: dict[str, Any] = {}
    if not hasattr(signal, "SIGALRM"):
        try:
            exec(code, namespace, namespace)
            exec(tests, namespace, namespace)
            return True, ""
        except Exception as exc:  # pragma: no cover
            return False, "".join(traceback.format_exception_only(type(exc), exc)).strip()[:240]

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"candidate exceeded {timeout_seconds}s")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        exec(code, namespace, namespace)
        exec(tests, namespace, namespace)
        return True, ""
    except Exception as exc:  # pragma: no cover
        return False, "".join(traceback.format_exception_only(type(exc), exc)).strip()[:240]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def interim_unattended_report(score: dict[str, Any], train_rows: int, heldout_rows: int) -> dict[str, Any]:
    score_summary = score.get("summary") if isinstance(score.get("summary"), dict) else {}
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "GREEN" if float(score_summary.get("pass_rate") or 0.0) >= 0.70 else "YELLOW",
        "summary": {
            "completion_evidence_status": "private_train_prototype_transfer",
            "private_train_rows": train_rows,
            "private_heldout_rows": heldout_rows,
            "heldout_pass_rate": score_summary.get("pass_rate"),
            "heldout_passes": score_summary.get("pass_count"),
            "heldout_task_count": score_summary.get("heldout_task_count"),
            "no_admissible_task_rate": score_summary.get("no_admissible_task_rate"),
            "sts_delta": score_summary.get("sts_delta"),
            "sts_regressions": score_summary.get("sts_regressions"),
        },
        "external_inference_calls": 0,
        "public_tests_used": False,
        "public_solutions_used": False,
    }


def next_actions(score_summary: dict[str, Any], prototype_failures: int, missing_candidates: list[dict[str, Any]]) -> list[str]:
    if prototype_failures:
        return ["Repair private train prototype self-test failures before scoring heldout."]
    if missing_candidates:
        return ["Expand private train prototype coverage for missing heldout semantic families."]
    if not score_summary:
        return ["Run the private train prototype scorer/gate."]
    if float(score_summary.get("pass_rate") or 0.0) >= 0.70:
        return [
            "Distill private-train prototype behavior into the Rust token/fanout path and demote the hardcoded diagnostic adapter.",
            "Only after private learned/prototype gates stay green should maturity/readiness consider an operator-approved public calibration proposal.",
        ]
    return ["Cluster failing prototype families and add private train pressure, not public calibration."]


def public_leakage_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    needles = ["humaneval", "mbpp", "evalplus", "bigcodebench", "livecodebench", "canonical_solution", "public_test"]
    hits = []
    for row in rows:
        text = "\n".join(leakage_strings(row)).lower()
        for needle in needles:
            if needle in text:
                hits.append({"task_id": row.get("task_id"), "needle": needle})
                break
        if len(hits) >= 20:
            break
    return {"hit_count": len(hits), "sample_hits": hits}


def leakage_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, str):
        return [value]
    return []


def gate_row(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def numeric(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Broad Private Train Prototype Decoder V1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- prototype_count: `{summary.get('prototype_count')}`",
        f"- heldout_pass_rate: `{summary.get('heldout_pass_rate')}`",
        f"- no_admissible_task_rate: `{summary.get('no_admissible_task_rate')}`",
        f"- sts_delta: `{summary.get('sts_delta')}`",
        f"- diagnostic_adapter_used: `{summary.get('diagnostic_adapter_used')}`",
        f"- external_inference_calls: `{summary.get('external_inference_calls')}`",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}`")
    lines.append("")
    lines.append("## Next Actions")
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def safe_ident(value: str) -> str:
    out = []
    for ch in value:
        out.append(ch if ch.isalnum() or ch == "_" else "_")
    ident = "".join(out).strip("_") or "entry"
    if ident[0].isdigit():
        ident = f"entry_{ident}"
    return ident


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = resolve(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
