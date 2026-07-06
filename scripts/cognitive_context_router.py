"""Build governed cognitive context spaces on top of STS.

This lane gives Theseus dedicated trainable spaces for private planning
summaries, response drafting, response review, durable artifact work, memory,
and final visible reports. It is intentionally not a public benchmark solver:
it never copies public answers or hidden tests, and it never treats private
planning as promotion evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_POLICY = ROOT / "configs" / "cognitive_context_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--base-sts", default="data/sts_learning/sts_code_streams_seed14.jsonl")
    parser.add_argument("--out-data", default="")
    parser.add_argument("--merged-out-data", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args()

    policy = read_json(resolve(args.policy), {})
    reporting = object_field(policy, "reporting")
    out_data = resolve(args.out_data or reporting.get("out_data") or "data/sts_learning/cognitive_context_spaces_seed14.jsonl")
    merged_out = resolve(args.merged_out_data or reporting.get("merged_sts_out") or "data/sts_learning/sts_code_context_spaces_seed14.jsonl")
    out_report = resolve(args.out or reporting.get("report") or "reports/cognitive_context_router.json")
    markdown_out = resolve(args.markdown_out or reporting.get("markdown") or "reports/cognitive_context_router.md")

    base_rows = read_jsonl(resolve(args.base_sts))
    rows = build_rows(policy)
    write_jsonl(out_data, rows)
    write_jsonl(merged_out, base_rows + rows)

    report = build_report(
        policy=policy,
        base_rows=base_rows,
        rows=rows,
        out_data=out_data,
        merged_out=merged_out,
        policy_path=resolve(args.policy),
        base_path=resolve(args.base_sts),
    )
    write_json(out_report, report)
    write_text(markdown_out, render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def build_rows(policy: dict[str, Any]) -> list[dict[str, Any]]:
    streams = [str(item) for item in policy.get("streams", []) if str(item)]
    learning = read_json(REPORTS / "learning_scoreboard.json", {})
    watchdog = read_json(REPORTS / "autonomy_watchdog.json", {})
    personality = read_json(REPORTS / "personality_context_last.json", {})
    memory = read_json(REPORTS / "context_packet_ledger.json", {})
    vcm = read_json(REPORTS / "virtual_context_memory_probe.json", {})
    guidance = read_json(REPORTS / "architecture_guidance_loop.json", {})
    taming = read_json(REPORTS / "deterministic_taming_stack.json", {})
    residual = read_json(REPORTS / "code_residual_curriculum.json", {})

    scoreboard_context = (
        f"public_pass={get_path(learning, ['public_transfer', 'real_public_task_pass_rate'], 'unknown')} "
        f"floor={get_path(learning, ['public_transfer', 'required_floor'], 'unknown')} "
        f"promotion_allowed={get_path(learning, ['promotion', 'promotion_allowed'], False)} "
        f"blockers={','.join(get_path(learning, ['promotion', 'honest_blockers'], []) or [])}"
    )
    runtime_context = (
        f"watchdog={watchdog.get('trigger_state', 'missing')} "
        f"phase={get_path(watchdog, ['summary', 'sparkstream_phase'], get_path(learning, ['operational_health', 'sparkstream_phase'], 'unknown'))} "
        f"active_wall={get_path(watchdog, ['summary', 'active_frontier_wall'], False)}"
    )
    personality_context = (
        f"personality={personality.get('status', 'missing')} "
        f"cards={get_path(personality, ['summary', 'selected_cards'], 0)} "
        f"hard_invariants={get_path(personality, ['summary', 'hard_safety_invariants'], 0)}"
    )
    memory_context = (
        f"context_packets={get_path(memory, ['active_packet_count'], get_path(memory, ['summary', 'active_packet_count'], 0))} "
        f"vcm={vcm.get('trigger_state', 'missing')} "
        f"vcm_pages={get_path(vcm, ['summary', 'semantic_pages'], 0)} "
        f"vcm_faults={get_path(vcm, ['summary', 'semantic_fault_count'], 0)} "
        f"stale_lanes={len(get_path(learning, ['stale_or_superseded_lanes'], []) or [])}"
    )
    wall_context = (
        f"wall={get_path(guidance, ['diagnosis', 'wall'], 'unknown')} "
        f"dominant_residual={get_path(guidance, ['diagnosis', 'dominant_residual'], 'unknown')} "
        f"teacher={get_path(guidance, ['teacher', 'status'], 'not_requested')}"
    )
    rule_context = (
        f"taming={taming.get('trigger_state', 'missing')} "
        f"hard_failures={get_path(taming, ['summary', 'hard_failure_count'], 'unknown')} "
        f"residual_private_rows={get_path(residual, ['summary', 'private_row_count'], 0)}"
    )

    scenarios = [
        {
            "task_id": "cognitive_sleep_readiness_review",
            "split": "eval",
            "context": f"User is sleeping soon; verify readiness without hiding the honest wall. {scoreboard_context}; {runtime_context}",
            "internal": "plan_tags=readiness,truth,promotion_blocked; inspect watchdog, scoreboard, active profile, and rotation",
            "artifact": "update reports/cognitive_context_router.json and leave durable next action for active code_lm_closure run",
            "draft": "System is alive, code learning is running, promotion remains blocked until public transfer clears the floor.",
            "review": "passes truth check: does not claim ASI, does not promote private pressure, reports active wall",
            "visible": "Report operational readiness, active benchmark pressure, and the remaining public-transfer blocker.",
        },
        {
            "task_id": "cognitive_code_wall_private_training",
            "split": "train",
            "context": f"Close public transfer honestly. {scoreboard_context}; {wall_context}; {rule_context}",
            "internal": "plan_tags=edge_case,private_curriculum,full_body_decoder; train private residual repairs then rerun public calibration",
            "artifact": "persist D:/ProjectTheseus/training_data/residual_code_curriculum/private_train/residual_code_lm_tasks.jsonl as private-only pressure",
            "draft": "Use public failures only as residual categories, then train on private generated lookalikes.",
            "review": "passes anti-cheat: no public prompt, no public solution, no hidden public test copied",
            "visible": "Next work is full Code LM Closure on private residual data, followed by same-seed public calibration.",
        },
        {
            "task_id": "cognitive_teacher_architecture_guidance",
            "split": "train",
            "context": f"Teacher should guide architecture only. {wall_context}; {scoreboard_context}",
            "internal": "plan_tags=teacher_sparse,proposal_only,experiment_spec; ask for decoder/verifier diagnosis only if local progress stalls",
            "artifact": "write reports/teacher_architecture_guidance_prompt.md with hashes, counts, residual classes, and no answers",
            "draft": "The teacher may propose an architecture experiment, but local measured evals decide adoption.",
            "review": "teacher apply mode forbidden; answer distillation forbidden; public benchmark solving forbidden",
            "visible": "Teacher remains a sparse architecture critic, not a solver or distillation source.",
        },
        {
            "task_id": "cognitive_artifact_workspace_build",
            "split": "train",
            "context": f"Maintain durable invention artifacts while the learner runs. {memory_context}; {personality_context}",
            "internal": "plan_tags=artifact_workspace,genesis,memory; capture reports and summaries, not raw noise",
            "artifact": "ingest readiness, taming, guidance, residual, and STS reports into context packets and Genesis artifacts",
            "draft": "Long-horizon work belongs in the artifact workspace, not in the visible response stream.",
            "review": "visible response should cite durable reports and avoid exposing raw private scratchpad text",
            "visible": "Keep artifact memory compact, provenance-bearing, and separate from user-facing answers.",
        },
        {
            "task_id": "cognitive_mouthpiece_review",
            "split": "eval",
            "context": f"Construct a response from private planning while preserving truth and personality. {personality_context}; {scoreboard_context}",
            "internal": "plan_tags=mouthpiece,review,grounded_status; decide what is safe and useful to say",
            "artifact": "record response review result as an audit summary if the message changes system state",
            "draft": "I checked readiness, implemented context spaces, and left the loop running.",
            "review": "ensure final does not reveal private monologue, overstate learning, or imply public promotion",
            "visible": "Send a concise grounded status with implemented files, verification, and honest blocker.",
        },
        {
            "task_id": "cognitive_benchmark_rotation",
            "split": "train",
            "context": f"Keep benchmark pressure rotating without cheating. {runtime_context}; {scoreboard_context}",
            "internal": "plan_tags=rotation,evalplus,mbpp,humaneval,bigcodebench; rotate after stall threshold and load transfer artifacts",
            "artifact": "ensure watchdog checks code-family stall and profile exports residual transfer artifacts",
            "draft": "If EvalPlus remains below floor past the threshold, rotate within code-family and return with transfer loaded.",
            "review": "rotation is curriculum pressure, not score shopping; promotion still requires honest public calibration",
            "visible": "Benchmark rotation should continue inside the code family while preserving public-score semantics.",
        },
    ]

    return [make_row(policy, streams, scenario, index) for index, scenario in enumerate(scenarios)]


def make_row(policy: dict[str, Any], streams: list[str], scenario: dict[str, str], index: int) -> dict[str, Any]:
    input_streams = {
        "system_policy_stream": "local-only; no public answers; no teacher solving; visible output requires review",
        "context_stream": scenario["context"],
        "personality_stream": "truth before compliance; agency before convenience; preserve oversight and anti-drift",
        "memory_context_stream": "use compact durable report summaries; avoid stale raw logs",
    }
    target_streams = {
        "internal_monologue_stream": scenario["internal"],
        "artifact_workspace_stream": scenario["artifact"],
        "mouthpiece_draft_stream": scenario["draft"],
        "response_review_stream": scenario["review"],
        "visible_report_stream": scenario["visible"],
    }
    return {
        "policy": "project_theseus_cognitive_context_sts_row_v1",
        "source_id": "local_cognitive_context_router",
        "task_id": scenario["task_id"],
        "split": scenario["split"],
        "row_index": index,
        "streams": streams,
        "input_streams": input_streams,
        "target_streams": target_streams,
        "public_benchmark": False,
        "benchmark_evidence_level": "private_cognitive_context_train_or_eval_only",
        "promotion_evidence": False,
        "visibility": object_field(policy, "visibility"),
        "causal_contract": {
            **object_field(policy, "causal_contract"),
            "visible_report_depends_on": ["mouthpiece_draft_stream", "response_review_stream"],
            "mouthpiece_depends_on": ["internal_monologue_stream", "context_stream", "personality_stream"],
            "artifact_workspace_depends_on": ["memory_context_stream", "context_stream"],
        },
        "provenance": {
            "origin": "local_reports_and_policy_summaries",
            "public_benchmark_solution_chars": 0,
            "public_test_chars": 0,
            "external_inference_calls": 0,
        },
    }


def build_report(
    *,
    policy: dict[str, Any],
    base_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    out_data: Path,
    merged_out: Path,
    policy_path: Path,
    base_path: Path,
) -> dict[str, Any]:
    security = object_field(policy, "security")
    min_rows = int(get_path(policy, ["gates", "min_context_rows"], 4) or 4)
    min_spaces = int(get_path(policy, ["gates", "min_context_space_count"], 6) or 6)
    all_streams = [str(item) for item in policy.get("streams", []) if str(item)]
    target_spaces = set()
    hard_issues = []
    for row in rows:
        target_spaces.update((row.get("target_streams") or {}).keys())
        hard_issues.extend(validate_row(row, policy))
    forbidden_visible_hits = [
        issue
        for issue in hard_issues
        if issue.get("type") in {"forbidden_visible_pattern", "raw_internal_exposed"}
    ]
    public_solution_chars = sum(int(get_path(row, ["provenance", "public_benchmark_solution_chars"], 0) or 0) for row in rows)
    public_test_chars = sum(int(get_path(row, ["provenance", "public_test_chars"], 0) or 0) for row in rows)
    gates = [
        gate("policy_loaded", policy.get("policy") == "project_theseus_cognitive_context_policy_v1", policy.get("policy")),
        gate("base_sts_rows_loaded", bool(base_rows), f"base_rows={len(base_rows)}"),
        gate("context_rows_written", len(rows) >= min_rows, f"rows={len(rows)} min={min_rows}"),
        gate("context_space_count", len(all_streams) >= min_spaces, f"spaces={all_streams}"),
        gate("stream_schema_present", all(set(all_streams).issubset(set(row.get("streams", []))) for row in rows), f"streams={len(all_streams)}"),
        gate("visible_report_requires_review", all(visible_requires_review(row) for row in rows), "visible report gated by mouthpiece and review streams"),
        gate("raw_internal_monologue_not_visible", not forbidden_visible_hits, forbidden_visible_hits[:3]),
        gate("public_benchmark_solutions_absent", public_solution_chars == 0 and public_test_chars == 0, f"solution_chars={public_solution_chars} test_chars={public_test_chars}"),
        gate("merged_sts_written", merged_out.exists() and len(base_rows) + len(rows) > len(base_rows), rel(merged_out)),
        gate("external_inference_zero", True, "deterministic local context routing"),
    ]
    trigger_state = "GREEN" if all(item["passed"] for item in gates) and not hard_issues else "RED"
    return {
        "policy": "project_theseus_cognitive_context_router_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "purpose": "Governed context-space routing for private planning summaries, mouthpiece draft, review, artifact workspace, memory, and visible report streams.",
        "config": rel(policy_path),
        "artifacts": {
            "base_sts": rel(base_path),
            "context_sts_rows": rel(out_data),
            "merged_sts_rows": rel(merged_out),
        },
        "summary": {
            "base_row_count": len(base_rows),
            "context_row_count": len(rows),
            "merged_row_count": len(base_rows) + len(rows),
            "stream_count": len(all_streams),
            "target_context_space_count": len(target_spaces),
            "target_context_spaces": sorted(target_spaces),
            "train_row_count": sum(1 for row in rows if row.get("split") == "train"),
            "eval_row_count": sum(1 for row in rows if row.get("split") == "eval"),
            "visible_report_requires_review": all(visible_requires_review(row) for row in rows),
            "raw_chain_of_thought_exposure": policy.get("raw_chain_of_thought_exposure"),
            "public_benchmark_solutions_included": False,
            "public_tests_included": False,
            "external_inference_calls": 0,
        },
        "issues": hard_issues,
        "gates": gates,
        "score_semantics": "private cognitive context STS substrate only; not public benchmark promotion evidence",
        "external_inference_calls": 0,
    }


def validate_row(row: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    issues = []
    security = object_field(policy, "security")
    max_cell_chars = int(security.get("max_cell_chars") or 1200)
    max_total = int(security.get("max_total_chars_per_row") or 5000)
    total_chars = 0
    streams = row.get("input_streams") if isinstance(row.get("input_streams"), dict) else {}
    targets = row.get("target_streams") if isinstance(row.get("target_streams"), dict) else {}
    for stream, text in {**streams, **targets}.items():
        value = str(text or "")
        total_chars += len(value)
        if len(value) > max_cell_chars:
            issues.append(issue("cell_char_budget", "hard", f"{row.get('task_id')}:{stream}:{len(value)}"))
    if total_chars > max_total:
        issues.append(issue("row_char_budget", "hard", f"{row.get('task_id')}:{total_chars}"))
    visible = str(targets.get("visible_report_stream") or "").lower()
    internal = str(targets.get("internal_monologue_stream") or "").lower()
    if "internal_monologue" in visible or "private planning" in visible:
        issues.append(issue("raw_internal_exposed", "hard", str(row.get("task_id"))))
    for pattern in security.get("blocked_visible_patterns", []):
        if str(pattern).lower() in visible:
            issues.append(issue("forbidden_visible_pattern", "hard", f"{row.get('task_id')}:{pattern}"))
    if "solution" in internal and row.get("public_benchmark"):
        issues.append(issue("public_solution_leak_risk", "hard", str(row.get("task_id"))))
    return issues


def visible_requires_review(row: dict[str, Any]) -> bool:
    contract = object_field(row, "causal_contract")
    deps = contract.get("visible_report_depends_on")
    if not isinstance(deps, list):
        return False
    return {"mouthpiece_draft_stream", "response_review_stream"}.issubset(set(str(item) for item in deps))


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    return "\n".join(
        [
            "# Cognitive Context Router",
            "",
            f"- Trigger: {report.get('trigger_state')}",
            f"- Context rows: {summary.get('context_row_count')}",
            f"- Merged STS rows: {summary.get('merged_row_count')}",
            f"- Spaces: {', '.join(summary.get('target_context_spaces') or [])}",
            f"- Visible requires review: {summary.get('visible_report_requires_review')}",
            f"- Raw chain exposure: {summary.get('raw_chain_of_thought_exposure')}",
            f"- Public solutions/tests included: {summary.get('public_benchmark_solutions_included')} / {summary.get('public_tests_included')}",
            "",
            "This is private context-space training pressure, not public benchmark promotion evidence.",
            "",
        ]
    )


def issue(kind: str, severity: str, detail: str) -> dict[str, Any]:
    return {"type": kind, "severity": severity, "detail": detail}


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def object_field(value: Any, key: str) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get(key), dict):
        return value[key]
    return {}


def get_path(value: Any, path: list[Any], default: Any = None) -> Any:
    cur = value
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
