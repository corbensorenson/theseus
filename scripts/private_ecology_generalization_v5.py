#!/usr/bin/env python3
"""Private v5 task ecology for post-v4 generalization.

This lane is for the state where v4 private learned transfer is green but the
next public calibration remains operator-locked. It expands private pressure
away from benchmark-like code tasks into companion/operator workflows: memory,
tool transcripts, file/storage manifests, device routing, spatial metadata,
and long-horizon project plans.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_rows import training_data_path  # noqa: E402
from code_residual_curriculum import verify_private_solution_rows  # noqa: E402


POLICY = "project_theseus_private_ecology_generalization_v5"
CARD_ID = "private_ecology_generalization_v5"
EVIDENCE_LEVEL = "private_ecology_generalization_v5_generated_only"
CONTRACT_POLICY = "project_theseus_decoder_contract_v5_private_ecology_generalization"
TRAIN_DEFAULT = training_data_path(
    "high_transfer",
    "private_train",
    "private_ecology_generalization_v5_code_lm_tasks.jsonl",
)
HELDOUT_DEFAULT = training_data_path(
    "high_transfer",
    "private_eval",
    "private_ecology_generalization_v5_heldout_code_lm_tasks.jsonl",
)
DEFAULT_PACKET = ROOT / "reports" / "public_calibration_readiness_packet.json"
DEFAULT_RUNNER = ROOT / "reports" / "operator_bounded_public_calibration_dry_run.json"
DEFAULT_LOCK = ROOT / "reports" / "public_calibration_operator_lock.flag"
DEFAULT_POST_V4_PUBLIC = ROOT / "reports" / "real_code_benchmark_graduation_post_v4_seed23_5x32.json"
DEFAULT_QUEUE = ROOT / "reports" / "private_ecology_generalization_v5_queue.jsonl"

FAMILIES = (
    "project_memory_contracts",
    "tool_transcript_contracts",
    "file_storage_contracts",
    "device_route_contracts",
    "long_horizon_plan_contracts",
    "spatial_operator_contracts",
)


@dataclass(frozen=True)
class Template:
    family: str
    category: str
    prompt: str
    body: str
    tests: Callable[[str, int], str]
    return_shape: str
    type_family: str
    required_constructs: tuple[str, ...]
    tags: tuple[str, ...]
    visible_arg_count_hint: int = 1
    argument_roles: dict[str, str] | None = None
    semantic_family: str = ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=1800)
    parser.add_argument("--heldout-rows", type=int, default=720)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--private-train-out", default=TRAIN_DEFAULT)
    parser.add_argument("--private-heldout-out", default=HELDOUT_DEFAULT)
    parser.add_argument("--packet", default=rel(DEFAULT_PACKET))
    parser.add_argument("--operator-runner", default=rel(DEFAULT_RUNNER))
    parser.add_argument("--operator-lock", default=rel(DEFAULT_LOCK))
    parser.add_argument("--post-v4-public-result", default=rel(DEFAULT_POST_V4_PUBLIC))
    parser.add_argument("--queue-out", default=rel(DEFAULT_QUEUE))
    parser.add_argument("--out", default="reports/private_ecology_generalization_v5.json")
    parser.add_argument("--markdown-out", default="reports/private_ecology_generalization_v5.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = resolve(args.packet)
    runner_path = resolve(args.operator_runner)
    lock_path = resolve(args.operator_lock)
    post_v4_public_path = resolve(args.post_v4_public_result)
    packet = read_json(packet_path, {})
    runner = read_json(runner_path, {})
    preflight = preflight_state(packet, runner, lock_path, post_v4_public_path)
    templates = template_bank()
    train_rows = build_rows(
        templates,
        row_count=max(1200, int(args.train_rows)),
        split="train",
        seed=int(args.seed),
        id_offset=0,
    )
    heldout_rows = build_rows(
        templates,
        row_count=max(480, int(args.heldout_rows)),
        split="heldout",
        seed=int(args.seed) + 300_000,
        id_offset=3_000_000,
    )
    train_path = resolve(args.private_train_out)
    heldout_path = resolve(args.private_heldout_out)
    write_jsonl(train_path, train_rows)
    write_jsonl(heldout_path, heldout_rows)
    queue = work_queue(train_path, heldout_path, args)
    write_jsonl(resolve(args.queue_out), queue)

    train_check = verify_private_solution_rows(train_rows, max_failures=24)
    heldout_check = verify_private_solution_rows(heldout_rows, max_failures=24)
    train_family_counts = Counter(str(row.get("broad_private_family_v1")) for row in train_rows)
    heldout_family_counts = Counter(str(row.get("broad_private_family_v1")) for row in heldout_rows)
    train_categories = Counter(str(row.get("category")) for row in train_rows)
    heldout_categories = Counter(str(row.get("category")) for row in heldout_rows)
    leakage = public_leakage_scan(train_rows + heldout_rows)
    gates = [
        gate("private_ecology_safe_to_generate", preflight["private_ecology_safe_to_generate"], preflight),
        gate("no_post_v4_public_result_exists", not post_v4_public_path.exists(), rel(post_v4_public_path)),
        gate("private_train_rows_ge_1200", len(train_rows) >= 1200, len(train_rows)),
        gate("private_heldout_rows_ge_480", len(heldout_rows) >= 480, len(heldout_rows)),
        gate("required_family_count", set(train_family_counts) == set(FAMILIES), dict(train_family_counts)),
        gate("heldout_required_family_count", set(heldout_family_counts) == set(FAMILIES), dict(heldout_family_counts)),
        gate("category_diversity_ge_12", len(train_categories) >= 12 and len(heldout_categories) >= 12, {
            "train_categories": len(train_categories),
            "heldout_categories": len(heldout_categories),
        }),
        gate("private_train_solution_tests_pass", train_check["failure_count"] == 0, train_check),
        gate("private_heldout_solution_tests_pass", heldout_check["failure_count"] == 0, heldout_check),
        gate("public_data_leakage_zero", leakage["hit_count"] == 0, leakage),
        gate("queue_written_with_no_public_run", len(queue) >= 4, {"queue": rel(resolve(args.queue_out)), "items": len(queue)}),
        gate("external_inference_zero", True, 0),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "RED"
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": trigger_state,
        "purpose": "Private-only post-v4 task ecology expansion while public calibration is operator-locked.",
        "inputs": {
            "seed": int(args.seed),
            "template_count": len(templates),
            "public_benchmark_inputs_read": False,
            "public_prompts_used": False,
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_score_labels_used": False,
            "packet": rel(packet_path),
            "operator_runner": rel(runner_path),
            "operator_lock": rel(lock_path),
            "post_v4_public_result": rel(post_v4_public_path),
        },
        "outputs": {
            "private_train_jsonl": rel(train_path),
            "private_heldout_jsonl": rel(heldout_path),
            "queue": rel(resolve(args.queue_out)),
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
        },
        "summary": {
            "private_train_row_count": len(train_rows),
            "private_heldout_row_count": len(heldout_rows),
            "family_train_row_counts": dict(sorted(train_family_counts.items())),
            "family_heldout_row_counts": dict(sorted(heldout_family_counts.items())),
            "category_train_count": len(train_categories),
            "category_heldout_count": len(heldout_categories),
            "private_train_solution_failures": train_check["failure_count"],
            "private_heldout_solution_failures": heldout_check["failure_count"],
            "public_data_leakage_hit_count": leakage["hit_count"],
            "queue_item_count": len(queue),
            "packet_ready": preflight["packet_ready"],
            "private_ecology_safe_to_generate": preflight["private_ecology_safe_to_generate"],
            "operator_lock_active": preflight["operator_lock_active"],
            "post_v4_public_result_exists": post_v4_public_path.exists(),
            "external_inference_calls": 0,
            "score_semantics": "private synthetic task-ecology pressure only; not public calibration",
        },
        "families": family_reports(train_rows, heldout_rows),
        "gates": gates,
        "next_actions": [
            "use the queue artifact as the next private-only autopilot target while the public calibration lock remains active",
            "fan out a small heldout smoke before a full v5 heldout run",
            "teach/rank reusable workflows from private train rows; do not train on public prompts, tests, traces, or scores",
            "keep the guarded public one-shot path locked until explicit operator approval",
        ],
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def preflight_state(packet: dict[str, Any], runner: dict[str, Any], lock_path: Path, post_v4_public_path: Path) -> dict[str, Any]:
    packet_summary = object_field(packet, "summary")
    runner_summary = object_field(runner, "summary")
    packet_ready = bool(
        packet.get("policy") == "project_theseus_public_calibration_readiness_packet_v1"
        and packet.get("mode") == "post_distillation_v4_operator_review"
        and packet.get("trigger_state") == "GREEN"
        and packet.get("technical_ready_for_one_bounded_public_calibration") is True
        and packet.get("public_calibration_allowed") is False
        and packet.get("operator_lock_active") is True
    )
    packet_public_locked = bool(
        packet.get("policy") == "project_theseus_public_calibration_readiness_packet_v1"
        and packet.get("mode") == "post_distillation_v4_operator_review"
        and packet.get("public_calibration_allowed") is False
        and packet.get("operator_lock_active") is True
    )
    runner_ready = bool(
        runner.get("policy") == "project_theseus_operator_bounded_public_calibration_v1"
        and runner.get("trigger_state") == "GREEN"
        and runner_summary.get("ready_for_operator_approval") is True
        and runner_summary.get("executed") is False
        and runner_summary.get("output_exists_after") is False
    )
    runner_not_executed = bool(
        runner_summary.get("executed") is False
        and runner_summary.get("output_exists_after") is False
    )
    return {
        "packet_ready": packet_ready,
        "runner_ready": runner_ready,
        "packet_public_locked": packet_public_locked,
        "runner_not_executed": runner_not_executed,
        "operator_lock_active": lock_path.exists(),
        "post_v4_public_result_exists": post_v4_public_path.exists(),
        "private_ecology_safe_to_generate": packet_public_locked and runner_not_executed and lock_path.exists() and not post_v4_public_path.exists(),
        "packet_mode": packet.get("mode"),
        "packet_trigger_state": packet.get("trigger_state"),
        "packet_public_calibration_allowed": packet.get("public_calibration_allowed"),
        "packet_summary": {
            "v4_learned_only_pass_rate": packet_summary.get("v4_learned_only_pass_rate"),
            "prototype_pass_count": packet_summary.get("prototype_pass_count"),
            "public_surface_task_count": packet_summary.get("public_surface_task_count"),
        },
        "runner_trigger_state": runner.get("trigger_state"),
        "runner_summary": {
            "ready_for_operator_approval": runner_summary.get("ready_for_operator_approval"),
            "executed": runner_summary.get("executed"),
            "output_exists_after": runner_summary.get("output_exists_after"),
        },
    }


def template_bank() -> list[Template]:
    return [
        Template("project_memory_contracts", "v5_memory_latest_by_project", "Return latest project note text by project id.", memory_latest_body(), tests_memory_latest, "dict", "project_memory", ("loop", "branch", "locals", "dict"), ("memory", "project_state"), 1, {"data": "note_records"}, "memory_state_tracking"),
        Template("project_memory_contracts", "v5_memory_open_action_rollup", "Return sorted open action labels grouped by owner.", action_rollup_body(), tests_action_rollup, "dict", "project_memory", ("loop", "branch", "locals", "dict", "collection_ops"), ("memory", "actions"), 1, {"data": "action_records"}, "action_memory_rollup"),
        Template("tool_transcript_contracts", "v5_tool_transcript_status", "Parse tool transcript lines into command/status records.", transcript_status_body(), tests_transcript_status, "list", "tool_transcript", ("loop", "branch", "locals", "parsing"), ("tool_use", "transcript"), 1, {"data": "transcript_text"}, "tool_status_parsing"),
        Template("tool_transcript_contracts", "v5_tool_error_clusters", "Cluster tool errors into timeout, permission, network, or other counts.", error_clusters_body(), tests_error_clusters, "dict", "tool_transcript", ("loop", "branch", "locals", "dict", "parsing"), ("tool_use", "errors"), 1, {"data": "error_lines"}, "tool_error_clustering"),
        Template("file_storage_contracts", "v5_storage_quota_select", "Select files under a quota by priority and stable name.", storage_quota_body(), tests_storage_quota, "list", "storage_manifest", ("loop", "branch", "locals", "collection_ops"), ("storage", "quota"), 2, {"data": "file_records", "other": "quota_bytes"}, "storage_selection"),
        Template("file_storage_contracts", "v5_storage_sync_plan", "Return upload/download/delete operations between local and remote manifests.", sync_plan_body(), tests_sync_plan, "list", "storage_manifest", ("loop", "branch", "locals", "dict", "collection_ops"), ("storage", "sync"), 2, {"data": "local_manifest", "other": "remote_manifest"}, "storage_sync_plan"),
        Template("device_route_contracts", "v5_device_route_worker", "Pick the best capable node for a worker request.", device_route_body(), tests_device_route, "str", "device_routing", ("loop", "branch", "locals", "selection"), ("device", "routing"), 2, {"data": "node_records", "other": "request"}, "capability_latency_routing"),
        Template("device_route_contracts", "v5_voice_output_route", "Pick the nearest speaker node for a voice response.", voice_route_body(), tests_voice_route, "str", "device_routing", ("loop", "branch", "locals", "selection"), ("voice", "routing"), 2, {"data": "node_records", "other": "room_hint"}, "voice_following_route"),
        Template("long_horizon_plan_contracts", "v5_plan_next_unblocked", "Return next unblocked tasks ordered by priority then id.", next_unblocked_body(), tests_next_unblocked, "list", "long_horizon_plan", ("loop", "branch", "locals", "collection_ops"), ("planning", "dependencies"), 1, {"data": "task_records"}, "dependency_planning"),
        Template("long_horizon_plan_contracts", "v5_plan_progress_digest", "Summarize done/blocked/open counts and next owners.", progress_digest_body(), tests_progress_digest, "dict", "long_horizon_plan", ("loop", "branch", "locals", "dict"), ("planning", "summary"), 1, {"data": "task_records"}, "project_progress_digest"),
        Template("spatial_operator_contracts", "v5_room_capability_summary", "Summarize device capabilities by room.", room_summary_body(), tests_room_summary, "dict", "spatial_operator", ("loop", "branch", "locals", "dict"), ("spatial", "rooms"), 1, {"data": "device_records"}, "room_capability_summary"),
        Template("spatial_operator_contracts", "v5_media_preview_index", "Return media ids matching tag and album filters sorted by date descending.", media_preview_body(), tests_media_preview, "list", "spatial_operator", ("loop", "branch", "locals", "collection_ops"), ("media", "preview"), 2, {"data": "media_records", "other": "query"}, "media_preview_retrieval"),
    ]


def build_rows(templates: list[Template], *, row_count: int, split: str, seed: int, id_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        template = templates[(index + seed) % len(templates)]
        variant = seed + index * 23
        rows.append(row_from_template(template, split=split, task_index=id_offset + index, variant=variant))
    return rows


def row_from_template(template: Template, *, split: str, task_index: int, variant: int) -> dict[str, Any]:
    entry = f"{template.category}_{task_index:07d}"
    tags = sorted({CARD_ID, split, template.family, template.category, *template.tags})
    return {
        "task_id": f"{CARD_ID}_{template.family}_{task_index:07d}",
        "source_task_id": f"{CARD_ID}_{split}_{variant:07d}",
        "card_id": CARD_ID,
        "source_id": f"local_generated_{CARD_ID}",
        "split": "train" if split == "train" else "eval",
        "category": template.category,
        "prompt": f"Private operator ecology contract: {template.prompt}",
        "entry_point": entry,
        "solution_expr": "",
        "solution_body": template.body,
        "tests": template.tests(entry, variant),
        "tags": tags,
        "broad_private_family_v1": template.family,
        "private_ecology_family_v5": template.family,
        "targeted_private_residual_family_v3": "private_ecology_generalization_v5",
        "residual_concept": template.semantic_family or template.category,
        "concept_residual_label": template.category,
        "metamorphic_properties": metamorphic_properties(template),
        "decoder_contract": decoder_contract(template),
        "benchmark_evidence_level": EVIDENCE_LEVEL,
        "public_benchmark": False,
        "public_benchmark_solutions_included": False,
        "public_tests_included": False,
        "public_prompts_included": False,
        "public_score_labels_included": False,
        "license_spdx": "CC0-1.0",
        "candidate_expression_eligible": False,
        "provenance": {
            "policy": POLICY,
            "family": template.family,
            "category": template.category,
            "variant": variant,
            "public_benchmark_answers_used": False,
            "public_tests_used": False,
            "public_prompts_used": False,
            "public_score_labels_used": False,
            "semantics": "private synthetic operator ecology pressure only",
        },
    }


def decoder_contract(template: Template) -> dict[str, Any]:
    return {
        "policy": CONTRACT_POLICY,
        "return_shape": template.return_shape,
        "type_family": template.type_family,
        "semantic_family": template.semantic_family or template.category,
        "visible_arg_count_hint": template.visible_arg_count_hint,
        "required_constructs": list(template.required_constructs),
        "residual_label_hint": template.category,
        "full_body_required": True,
        "guardrail_only": False,
        "feedback_weight": 1.5,
        "score_semantics": "private operator ecology generalization pressure only",
        "argument_roles": template.argument_roles or {"data": "primary_input"},
        "return_contract": {
            "shape": template.return_shape,
            "empty_or_invalid_behavior": "covered_by_private_v5_assertions",
            "must_preserve_container_shape": template.return_shape in {"list", "dict", "tuple"},
        },
        "generation_plan": {
            "policy": "private_ecology_solution_body -> reusable_operator_workflow_token_decoder -> heldout_contract_body",
            "skeleton_bias": list(template.required_constructs),
            "repair_strategy": "learn reusable operator workflow transforms instead of benchmark-specific adapters",
            "public_tests_used": False,
            "public_solutions_used": False,
        },
    }


def work_queue(train_path: Path, heldout_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    queue = [
        {
            "task_id": "private_ecology_v5_generate_rows",
            "kind": "private_curriculum_generation",
            "status": "completed",
            "artifact": rel(train_path),
            "public_calibration_allowed": False,
        },
        {
            "task_id": "private_ecology_v5_smoke_fanout",
            "kind": "private_code_lm_fanout_smoke",
            "status": "pending",
            "heldout": rel(heldout_path),
            "task_limit": 72,
            "public_calibration_allowed": False,
        },
        {
            "task_id": "private_ecology_v5_learned_only_gate",
            "kind": "private_learned_distillation_gate",
            "status": "pending",
            "prototype_pass_count_required": 0,
            "public_calibration_allowed": False,
        },
        {
            "task_id": "private_ecology_v5_overnight_readiness_refresh",
            "kind": "overnight_readiness_refresh",
            "status": "pending",
            "command": "python3 scripts/overnight_learning_readiness.py --out reports/overnight_learning_readiness.json --markdown-out reports/overnight_learning_readiness.md",
            "public_calibration_allowed": False,
        },
    ]
    return queue


def metamorphic_properties(template: Template) -> list[str]:
    common = {
        "project_memory_contracts": ["stable_latest_selection", "malformed_record_skip"],
        "tool_transcript_contracts": ["case_insensitive_error_classes", "malformed_line_skip"],
        "file_storage_contracts": ["stable_path_sort", "quota_or_hash_boundary"],
        "device_route_contracts": ["capability_filter_first", "latency_battery_tiebreak"],
        "long_horizon_plan_contracts": ["dependency_boundary", "priority_tiebreak"],
        "spatial_operator_contracts": ["room_label_boundary", "tag_filter_intersection"],
    }
    return common.get(template.family, ["private_ecology_generalization"])


def family_reports(train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    for family in FAMILIES:
        train = [row for row in train_rows if row.get("broad_private_family_v1") == family]
        heldout = [row for row in heldout_rows if row.get("broad_private_family_v1") == family]
        reports.append(
            {
                "family": family,
                "train_rows": len(train),
                "heldout_rows": len(heldout),
                "categories": sorted({str(row.get("category")) for row in train}),
                "decoder_contract_rows": sum(1 for row in train if isinstance(row.get("decoder_contract"), dict)),
            }
        )
    return reports


def public_leakage_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    needles = [
        "humaneval",
        "mbpp",
        "evalplus",
        "bigcodebench",
        "livecodebench",
        "canonical_solution",
        "public_test",
        "public prompt",
    ]
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
        out: list[str] = []
        for child in value:
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, str):
        return [value]
    return []


def memory_latest_body() -> str:
    return """latest = {}
for item in data or []:
    if not isinstance(item, dict):
        continue
    project = item.get("project")
    text = item.get("text")
    ts = item.get("ts", 0)
    if not project or text is None:
        continue
    old = latest.get(project)
    if old is None or ts >= old[0]:
        latest[project] = (ts, text)
return {key: value[1] for key, value in sorted(latest.items())}"""


def tests_memory_latest(entry: str, variant: int) -> str:
    return f"""assert {entry}([{{"project":"hive","text":"old","ts":1}}, {{"project":"hive","text":"new","ts":3}}, {{"project":"ios","text":"ship","ts":2}}]) == {{"hive":"new","ios":"ship"}}
assert {entry}([{{"project":"x","text":"a","ts":1}}, {{"project":"x","text":"b","ts":1}}]) == {{"x":"b"}}
assert {entry}([None, {{"text":"skip"}}, {{"project":"p","text":"ok"}}]) == {{"p":"ok"}}
assert {entry}([]) == {{}}"""


def action_rollup_body() -> str:
    return """out = {}
for item in data or []:
    if not isinstance(item, dict) or item.get("done"):
        continue
    owner = item.get("owner") or "unassigned"
    label = item.get("label")
    if not label:
        continue
    out.setdefault(owner, set()).add(str(label))
return {owner: sorted(labels) for owner, labels in sorted(out.items())}"""


def tests_action_rollup(entry: str, variant: int) -> str:
    return f"""rows = [{{"owner":"mac","label":"mlx","done":False}}, {{"owner":"mac","label":"mlx","done":False}}, {{"owner":"win","label":"cuda","done":False}}, {{"owner":"mac","label":"done","done":True}}]
assert {entry}(rows) == {{"mac":["mlx"],"win":["cuda"]}}
assert {entry}([{{"label":"a"}}, {{"owner":"b"}}, {{"owner":"b","label":"c","done":False}}]) == {{"b":["c"],"unassigned":["a"]}}
assert {entry}([]) == {{}}
assert {entry}([None, 3]) == {{}}"""


def transcript_status_body() -> str:
    return """rows = []
current = None
for raw in str(data or "").splitlines():
    line = raw.strip()
    if line.startswith("$"):
        current = line[1:].strip()
    elif line.startswith("=>") and current:
        status = line[2:].strip().lower() or "unknown"
        rows.append({"command": current, "status": status})
        current = None
return rows"""


def tests_transcript_status(entry: str, variant: int) -> str:
    return f"""assert {entry}("$ build\\n=> ok\\n$ test\\n=> fail") == [{{"command":"build","status":"ok"}}, {{"command":"test","status":"fail"}}]
assert {entry}("noise\\n$ probe\\n=> OK") == [{{"command":"probe","status":"ok"}}]
assert {entry}("$ orphan") == []
assert {entry}("") == []"""


def error_clusters_body() -> str:
    return """counts = {"network": 0, "other": 0, "permission": 0, "timeout": 0}
for item in data or []:
    text = str(item).lower()
    if "timeout" in text or "timed out" in text:
        counts["timeout"] += 1
    elif "permission" in text or "denied" in text:
        counts["permission"] += 1
    elif "network" in text or "connection" in text or "dns" in text:
        counts["network"] += 1
    elif text.strip():
        counts["other"] += 1
return counts"""


def tests_error_clusters(entry: str, variant: int) -> str:
    return f"""assert {entry}(["timeout waiting", "Permission denied", "dns failed", "weird"]) == {{"network":1,"other":1,"permission":1,"timeout":1}}
assert {entry}(["TIMED OUT", "connection reset"]) == {{"network":1,"other":0,"permission":0,"timeout":1}}
assert {entry}(["", None]) == {{"network":0,"other":1,"permission":0,"timeout":0}}
assert {entry}([]) == {{"network":0,"other":0,"permission":0,"timeout":0}}"""


def storage_quota_body() -> str:
    return """quota = int(other or 0)
items = []
for item in data or []:
    if not isinstance(item, dict):
        continue
    name = item.get("name")
    size = int(item.get("size") or 0)
    priority = int(item.get("priority") or 0)
    if not name or size <= 0 or size > quota:
        continue
    items.append((-priority, size, str(name)))
used = 0
picked = []
for _prio, size, name in sorted(items):
    if used + size <= quota:
        picked.append(name)
        used += size
return picked"""


def tests_storage_quota(entry: str, variant: int) -> str:
    return f"""files = [{{"name":"a","size":5,"priority":1}}, {{"name":"b","size":6,"priority":3}}, {{"name":"c","size":4,"priority":2}}]
assert {entry}(files, 10) == ["b","c"]
assert {entry}(files, 5) == ["c"]
assert {entry}([{{"name":"x","size":0}}, {{"size":1}}], 10) == []
assert {entry}([], 9) == []"""


def sync_plan_body() -> str:
    return """ops = []
local = data if isinstance(data, dict) else {}
remote = other if isinstance(other, dict) else {}
for path in sorted(set(local) | set(remote)):
    left = local.get(path)
    right = remote.get(path)
    if left is None:
        ops.append(("download", path))
    elif right is None:
        ops.append(("upload", path))
    elif left != right:
        ops.append(("upload", path))
return ops"""


def tests_sync_plan(entry: str, variant: int) -> str:
    return f"""assert {entry}({{"a":"1","b":"2"}}, {{"b":"2","c":"3"}}) == [("upload","a"),("download","c")]
assert {entry}({{"a":"new"}}, {{"a":"old"}}) == [("upload","a")]
assert {entry}({{}}, {{"r":"1"}}) == [("download","r")]
assert {entry}({{"same":"x"}}, {{"same":"x"}}) == []"""


def device_route_body() -> str:
    return """request = other if isinstance(other, dict) else {}
need = set(request.get("capabilities") or [])
best = None
for node in data or []:
    if not isinstance(node, dict):
        continue
    caps = set(node.get("capabilities") or [])
    if not need <= caps:
        continue
    if request.get("avoid_battery") and node.get("battery") is True:
        continue
    score = (float(node.get("latency_ms") or 999999), -float(node.get("memory_gb") or 0), str(node.get("name") or ""))
    if best is None or score < best[0]:
        best = (score, str(node.get("name") or ""))
return best[1] if best else ""
"""


def tests_device_route(entry: str, variant: int) -> str:
    return f"""nodes = [{{"name":"cpu","capabilities":["cpu"],"latency_ms":4}}, {{"name":"mlx","capabilities":["cpu","mlx"],"latency_ms":8,"memory_gb":32}}, {{"name":"fast","capabilities":["cpu","mlx"],"latency_ms":5,"memory_gb":16}}]
assert {entry}(nodes, {{"capabilities":["mlx"]}}) == "fast"
assert {entry}(nodes, {{"capabilities":["cuda"]}}) == ""
assert {entry}([{{"name":"bat","capabilities":["cpu"],"battery":True}}, {{"name":"plug","capabilities":["cpu"]}}], {{"capabilities":["cpu"],"avoid_battery":True}}) == "plug"
assert {entry}([], {{"capabilities":["cpu"]}}) == ""
"""


def voice_route_body() -> str:
    return """room = str(other or "")
best = None
for node in data or []:
    if not isinstance(node, dict) or not node.get("speaker"):
        continue
    closeness = 0 if str(node.get("room") or "") == room else 1
    confidence = -float(node.get("confidence") or 0)
    latency = float(node.get("latency_ms") or 999999)
    score = (closeness, confidence, latency, str(node.get("name") or ""))
    if best is None or score < best[0]:
        best = (score, str(node.get("name") or ""))
return best[1] if best else ""
"""


def tests_voice_route(entry: str, variant: int) -> str:
    return f"""nodes = [{{"name":"kitchen","room":"kitchen","speaker":True,"confidence":0.8}}, {{"name":"hall","room":"hall","speaker":True,"confidence":0.95}}]
assert {entry}(nodes, "kitchen") == "kitchen"
assert {entry}(nodes, "office") == "hall"
assert {entry}([{{"name":"mic","speaker":False}}], "x") == ""
assert {entry}([], "x") == ""
"""


def next_unblocked_body() -> str:
    return """done = {str(item.get("id")) for item in data or [] if isinstance(item, dict) and item.get("done")}
available = []
for item in data or []:
    if not isinstance(item, dict) or item.get("done"):
        continue
    deps = {str(dep) for dep in item.get("deps") or []}
    if deps <= done:
        available.append((-int(item.get("priority") or 0), str(item.get("id") or "")))
return [task_id for _prio, task_id in sorted(available) if task_id]"""


def tests_next_unblocked(entry: str, variant: int) -> str:
    return f"""tasks = [{{"id":"a","done":True}}, {{"id":"b","deps":["a"],"priority":2}}, {{"id":"c","deps":["x"],"priority":9}}, {{"id":"d","deps":[],"priority":1}}]
assert {entry}(tasks) == ["b","d"]
assert {entry}([{{"id":"x","done":False,"deps":[]}}]) == ["x"]
assert {entry}([{{"id":"x","done":True}}]) == []
assert {entry}([]) == []"""


def progress_digest_body() -> str:
    return """out = {"blocked": 0, "done": 0, "open": 0, "owners": []}
owners = set()
for item in data or []:
    if not isinstance(item, dict):
        continue
    if item.get("done"):
        out["done"] += 1
    elif item.get("blocked"):
        out["blocked"] += 1
    else:
        out["open"] += 1
        if item.get("owner"):
            owners.add(str(item.get("owner")))
out["owners"] = sorted(owners)
return out"""


def tests_progress_digest(entry: str, variant: int) -> str:
    return f"""rows = [{{"done":True}}, {{"blocked":True}}, {{"owner":"mac"}}, {{"owner":"ios"}}]
assert {entry}(rows) == {{"blocked":1,"done":1,"open":2,"owners":["ios","mac"]}}
assert {entry}([{{"owner":"x"}}, {{"owner":"x"}}]) == {{"blocked":0,"done":0,"open":2,"owners":["x"]}}
assert {entry}([None]) == {{"blocked":0,"done":0,"open":0,"owners":[]}}
assert {entry}([]) == {{"blocked":0,"done":0,"open":0,"owners":[]}}"""


def room_summary_body() -> str:
    return """rooms = {}
for node in data or []:
    if not isinstance(node, dict):
        continue
    room = str(node.get("room") or "unknown")
    rec = rooms.setdefault(room, {"devices": 0, "mics": 0, "speakers": 0})
    rec["devices"] += 1
    if node.get("mic"):
        rec["mics"] += 1
    if node.get("speaker"):
        rec["speakers"] += 1
return {room: rooms[room] for room in sorted(rooms)}"""


def tests_room_summary(entry: str, variant: int) -> str:
    return f"""nodes = [{{"room":"kitchen","mic":True}}, {{"room":"kitchen","speaker":True}}, {{"room":"office","mic":True,"speaker":True}}]
assert {entry}(nodes) == {{"kitchen":{{"devices":2,"mics":1,"speakers":1}},"office":{{"devices":1,"mics":1,"speakers":1}}}}
assert {entry}([{{"mic":True}}]) == {{"unknown":{{"devices":1,"mics":1,"speakers":0}}}}
assert {entry}([None]) == {{}}
assert {entry}([]) == {{}}"""


def media_preview_body() -> str:
    return """query = other if isinstance(other, dict) else {}
need_tags = set(query.get("tags") or [])
album = query.get("album")
hits = []
for item in data or []:
    if not isinstance(item, dict):
        continue
    if album and item.get("album") != album:
        continue
    tags = set(item.get("tags") or [])
    if not need_tags <= tags:
        continue
    hits.append((str(item.get("date") or ""), str(item.get("id") or "")))
return [media_id for _date, media_id in sorted(hits, reverse=True) if media_id]"""


def tests_media_preview(entry: str, variant: int) -> str:
    return f"""media = [{{"id":"1","album":"home","date":"2026-01-01","tags":["archive","presentation"]}}, {{"id":"2","album":"home","date":"2026-02-01","tags":["presentation"]}}, {{"id":"3","album":"work","date":"2026-03-01","tags":["presentation"]}}]
assert {entry}(media, {{"album":"home","tags":["presentation"]}}) == ["2","1"]
assert {entry}(media, {{"tags":["archive"]}}) == ["1"]
assert {entry}(media, {{"album":"missing"}}) == []
assert {entry}([], {{"tags":[]}}) == []"""


def object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else (default if default is not None else {})
    except Exception:
        return default if default is not None else {}


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


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Private Ecology Generalization V5",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Private train rows: {summary.get('private_train_row_count')}",
        f"- Private heldout rows: {summary.get('private_heldout_row_count')}",
        f"- Train categories: {summary.get('category_train_count')}",
        f"- Heldout categories: {summary.get('category_heldout_count')}",
        f"- Train solution failures: {summary.get('private_train_solution_failures')}",
        f"- Heldout solution failures: {summary.get('private_heldout_solution_failures')}",
        f"- Public-data leakage hits: {summary.get('public_data_leakage_hit_count')}",
        f"- Queue items: {summary.get('queue_item_count')}",
        "",
        "## Families",
    ]
    for row in report.get("families", []):
        lines.append(
            f"- `{row.get('family')}`: train {row.get('train_rows')}, heldout {row.get('heldout_rows')}, categories {len(row.get('categories') or [])}"
        )
    lines.extend(["", "No public benchmark prompts, tests, solutions, score labels, traces, or task ids are used."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
