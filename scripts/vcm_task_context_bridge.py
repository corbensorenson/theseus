"""Compile VCM task-context views for Theseus task families.

VCM is the shared memory/context layer for Theseus. This bridge turns the VCM
ledger, index, compiled context, and governance probes into explicit task-family
contracts that other systems can consume without inventing their own memory
rules.

It is deterministic and local-only: no teacher calls, no external inference, no
network fetches, no public calibration, no training row writes, and no fallback
returns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import vcm_consumer_abi


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_POLICY = ROOT / "configs" / "vcm_task_context_policy.json"
DEFAULT_INDEX = REPORTS / "virtual_context_memory_index.json"
DEFAULT_COMPILED = REPORTS / "virtual_context_compiled_context.json"
DEFAULT_PROBE = REPORTS / "virtual_context_memory_probe.json"
DEFAULT_STATUS = REPORTS / "virtual_context_memory_status.json"
DEFAULT_TRAINING = REPORTS / "virtual_context_memory_training_admission.json"
DEFAULT_CONSUMER_AUDIT = REPORTS / "virtual_context_memory_consumer_audit.json"
DEFAULT_RUNTIME = REPORTS / "vcm_runtime_claim_readiness.json"
DEFAULT_RELEASE = REPORTS / "vcm_release_conformance_audit.json"
DEFAULT_OUT = REPORTS / "vcm_task_context_bridge.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "vcm_task_context_bridge.md"
DEFAULT_CONTEXTS_OUT = REPORTS / "vcm_task_contexts.json"
DEFAULT_CONTEXT_GOVERNOR = REPORTS / "vcm_context_governor.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=rel(DEFAULT_POLICY))
    parser.add_argument("--index", default=rel(DEFAULT_INDEX))
    parser.add_argument("--compiled", default=rel(DEFAULT_COMPILED))
    parser.add_argument("--probe", default=rel(DEFAULT_PROBE))
    parser.add_argument("--status", default=rel(DEFAULT_STATUS))
    parser.add_argument("--training-admission", default=rel(DEFAULT_TRAINING))
    parser.add_argument("--consumer-audit", default=rel(DEFAULT_CONSUMER_AUDIT))
    parser.add_argument("--runtime-readiness", default=rel(DEFAULT_RUNTIME))
    parser.add_argument("--release-conformance", default=rel(DEFAULT_RELEASE))
    parser.add_argument("--vcm-governor", default=rel(DEFAULT_CONTEXT_GOVERNOR))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--contexts-out", default=rel(DEFAULT_CONTEXTS_OUT))
    args = parser.parse_args()

    started = time.perf_counter()
    report, contexts = build_report(args, started=started)
    write_json(resolve(args.out), report)
    write_json(resolve(args.contexts_out), contexts)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, *, started: float) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = read_json(resolve(args.policy))
    index = read_json(resolve(args.index))
    compiled = read_json(resolve(args.compiled))
    probe = read_json(resolve(args.probe))
    status = read_json(resolve(args.status))
    training = read_json(resolve(args.training_admission))
    consumer = read_json(resolve(args.consumer_audit))
    runtime = read_json(resolve(args.runtime_readiness))
    release = read_json(resolve(args.release_conformance))

    visible_pages = list_value(compiled.get("model_visible_pages"))
    protected_pages = select_protected_pages(visible_pages)
    defaults = dict_value(policy.get("defaults"))
    task_contexts = []
    gates = []
    blockers = []
    warnings = []

    for family in list_value(policy.get("task_families")):
        if not isinstance(family, dict):
            continue
        context = build_task_context(
            family=family,
            visible_pages=visible_pages,
            protected_pages=protected_pages,
            defaults=defaults,
            compiled=compiled,
            probe=probe,
            status=status,
            training=training,
            runtime=runtime,
        )
        task_contexts.append(context)
        if not context["ready"]:
            issue = {
                "task_family": context["task_family_id"],
                "label": context["label"],
                "priority": context["priority"],
                "reasons": context["blockers"],
            }
            if context["priority"] == "high":
                blockers.append(issue)
            else:
                warnings.append(issue)

    selected_pages = [page for context in task_contexts for page in list_value(context.get("selected_pages")) if isinstance(page, dict)]
    consumer_packet = vcm_consumer_abi.build_consumer_packet(
        consumer_id="vcm_task_context_bridge",
        purpose="task_context_compilation",
        read_set=[
            rel(resolve(args.vcm_governor)),
            rel(resolve(args.policy)),
            rel(resolve(args.index)),
            rel(resolve(args.compiled)),
        ],
        write_set=[rel(resolve(args.out)), rel(resolve(args.contexts_out))],
        authority_ceiling=["local_vcm_metadata_read", "task_context_packet_write"],
        permitted_uses=["task_context_compilation", "context_selection", "audit_replay"],
        governor_path=resolve(args.vcm_governor),
        semantic_index_path=resolve(args.index),
        context_refs=[
            {
                "kind": "semantic_address",
                "ref": page.get("address") or page.get("source_path"),
                "required": True,
                "exists": bool(page.get("address") or page.get("source_path")),
                "sha256": page.get("content_hash") or page.get("sha256") or "",
                "taint_labels": page.get("taints", []),
                "contradiction_refs": page.get("contradiction_refs", []),
            }
            for page in selected_pages
        ],
        taint_labels=sorted({str(taint) for page in selected_pages for taint in list_value(page.get("taints"))}),
        deletion_obligations=["invalidate_task_contexts_when_source_pages_are_revoked"],
        contradiction_refs=[
            str(ref)
            for page in selected_pages
            for ref in list_value(page.get("contradiction_refs"))
            if ref
        ],
        audit_refs=["scripts/vcm_task_context_bridge.py"],
    )

    gates.extend(system_gates(policy, probe, status, training, consumer, runtime, release, compiled, task_contexts))
    gates.append(gate("semantic_index_loaded", bool(list_value(index.get("pages"))), len(list_value(index.get("pages"))), "hard"))
    gates.extend(task_family_gates(task_contexts))
    gates.append(gate("no_high_priority_task_context_blockers", not blockers, blockers[:8], "hard"))
    gates.append(gate("medium_priority_task_context_warnings_recorded", isinstance(warnings, list), warnings[:8], "warning"))
    gates.append(gate("external_inference_zero", external_calls(probe, status, training, runtime, release) == 0, external_calls(probe, status, training, runtime, release), "hard"))
    gates.append(gate("public_training_rows_zero", public_training_rows(probe, status, training, runtime, release) == 0, public_training_rows(probe, status, training, runtime, release), "hard"))
    gates.append(gate("fallback_returns_zero", fallback_returns(probe, status, training, runtime, release) == 0, fallback_returns(probe, status, training, runtime, release), "hard"))
    gates.append(gate("vcm_consumer_abi_ready", bool(consumer_packet.get("ready")), consumer_packet.get("typed_faults"), "hard"))

    hard_failures = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    trigger_state = "GREEN" if not hard_failures and not blockers else "RED"
    if trigger_state == "GREEN" and warnings:
        trigger_state = "YELLOW"

    contexts_payload = {
        "policy": "project_theseus_vcm_task_contexts_v1",
        "created_utc": now(),
        "snapshot": str(compiled.get("snapshot") or ""),
        "task_context_count": len(task_contexts),
        "task_contexts": task_contexts,
        "vcm_consumer_abi_receipt": vcm_consumer_abi.compact_consumer_packet(consumer_packet),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "teacher_solving_calls": 0,
        "fallback_return_count": 0,
    }
    report = {
        "policy": "project_theseus_vcm_task_context_bridge_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "purpose": policy.get("purpose"),
        "inputs": {
            "policy": rel(resolve(args.policy)),
            "index": rel(resolve(args.index)),
            "compiled": rel(resolve(args.compiled)),
            "probe": rel(resolve(args.probe)),
            "status": rel(resolve(args.status)),
            "training_admission": rel(resolve(args.training_admission)),
            "consumer_audit": rel(resolve(args.consumer_audit)),
            "runtime_readiness": rel(resolve(args.runtime_readiness)),
            "release_conformance": rel(resolve(args.release_conformance)),
            "vcm_governor": rel(resolve(args.vcm_governor)),
        },
        "artifacts": {
            "contexts": rel(resolve(args.contexts_out)),
            "markdown": rel(resolve(args.markdown_out)),
        },
        "summary": {
            "task_family_count": len(task_contexts),
            "ready_task_family_count": len([row for row in task_contexts if row["ready"]]),
            "high_priority_task_family_count": len([row for row in task_contexts if row["priority"] == "high"]),
            "high_priority_ready_count": len([row for row in task_contexts if row["priority"] == "high" and row["ready"]]),
            "medium_priority_warning_count": len(warnings),
            "selected_page_count": sum(len(row["selected_pages"]) for row in task_contexts),
            "unique_selected_page_count": len({page["address"] for row in task_contexts for page in row["selected_pages"]}),
            "protected_page_count": len(protected_pages),
            "vcm_probe_state": probe.get("trigger_state"),
            "training_admission_state": training.get("trigger_state"),
            "consumer_audit_state": consumer.get("trigger_state"),
            "release_conformance_state": release.get("trigger_state"),
            "runtime_profile_claimed": bool(get_path(runtime, ["summary", "runtime_profile_claimed"], False)),
            "runtime_native_kv_claimed": bool(get_path(runtime, ["summary", "native_kv_cache_claimed"], False)),
            "vcm_consumer_abi_ready": consumer_packet.get("ready"),
            "vcm_consumer_abi_packet_id": consumer_packet.get("packet_id"),
            "snapshot": str(compiled.get("snapshot") or ""),
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "teacher_solving_calls": 0,
            "fallback_return_count": 0,
            "runtime_seconds": round(time.perf_counter() - started, 4),
        },
        "blockers": blockers,
        "warnings": warnings,
        "gates": gates,
        "task_contexts": task_contexts,
        "vcm_consumer_abi": consumer_packet,
        "integration_contract": {
            "load_before": [
                "operator_chat",
                "autonomy_cycle_decision",
                "training_launch",
                "teacher_call",
                "public_calibration_review",
                "hive_task_routing",
                "runtime_accelerator_routing"
            ],
            "write_after": [
                "accepted",
                "missed",
                "ignored",
                "operator"
            ],
            "not_claimed": [
                "native VCM-Runtime KV cache",
                "native prefix cache lifecycle",
                "public benchmark training",
                "teacher answer distillation",
                "fallback returns"
            ],
        },
        "external_inference_calls": 0,
    }
    return report, contexts_payload


def build_task_context(
    *,
    family: dict[str, Any],
    visible_pages: list[dict[str, Any]],
    protected_pages: list[dict[str, Any]],
    defaults: dict[str, Any],
    compiled: dict[str, Any],
    probe: dict[str, Any],
    status: dict[str, Any],
    training: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    max_pages = int(family.get("max_selected_pages") or defaults.get("max_selected_pages") or 12)
    min_pages = int(family.get("min_selected_pages") or defaults.get("min_selected_pages") or 3)
    terms = [str(term).lower() for term in list_value(family.get("query_terms")) if str(term)]
    preferred_lanes = {str(lane) for lane in list_value(family.get("preferred_lanes")) if str(lane)}
    scored: list[tuple[float, dict[str, Any]]] = []
    visible_addresses = {str(row.get("address") or "") for row in visible_pages}
    for page in visible_pages:
        if not isinstance(page, dict):
            continue
        score = page_score(page, terms=terms, preferred_lanes=preferred_lanes, visible_addresses=visible_addresses)
        if score > 0:
            scored.append((score, page))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("source_path") or ""), str(item[1].get("title") or "")))

    selected = dedupe_page_summaries([summarize_page(page, score=score) for score, page in scored[:max_pages]])
    if defaults.get("include_protected_minimum", True):
        selected = dedupe_page_summaries([*protected_pages, *selected])[:max_pages]

    guard_results = build_guard_results(
        family=family,
        defaults=defaults,
        compiled=compiled,
        probe=probe,
        status=status,
        training=training,
        runtime=runtime,
    )
    blockers = []
    if len(selected) < min_pages:
        blockers.append(f"selected_pages_below_minimum:{len(selected)}/{min_pages}")
    if any(page.get("model_visible") is not True for page in selected):
        blockers.append("selected_context_contains_non_model_visible_page")
    for row in guard_results:
        if row["severity"] == "hard" and not row["passed"]:
            blockers.append(str(row["guard"]))

    ready = not blockers
    selected_hash = stable_hash({"family": family.get("id"), "pages": [page.get("address") for page in selected]})
    return {
        "task_family_id": str(family.get("id") or "unknown"),
        "label": str(family.get("label") or family.get("id") or "unknown"),
        "priority": str(family.get("priority") or "medium"),
        "ready": ready,
        "selected_context_hash": selected_hash,
        "snapshot": str(compiled.get("snapshot") or ""),
        "query_terms": terms,
        "selected_pages": selected,
        "guard_results": guard_results,
        "writeback_contract": {
            "allowed_usage_event_kinds": list_value(family.get("required_writebacks")) or list_value(defaults.get("write_back_event_kinds")),
            "usage_events_path": rel(REPORTS / "virtual_context_memory_usage_events.jsonl"),
            "raw_text_storage": False,
            "training_requires_vcm_admission": True,
        },
        "fault_policy": {
            "capacity_faults_visible": True,
            "detail_faults_visible": True,
            "unsafe_fit_must_fail_closed": True,
            "fallback_returns_allowed": False,
        },
        "blockers": blockers,
        "notes": str(family.get("notes") or ""),
    }


def page_score(page: dict[str, Any], *, terms: list[str], preferred_lanes: set[str], visible_addresses: set[str]) -> float:
    haystack = " ".join(
        str(page.get(key) or "")
        for key in ["address", "lane", "type", "execution_class", "source_path", "title", "status"]
    ).lower()
    score = 0.0
    for term in terms:
        if term and term in haystack:
            score += 3.0 if " " in term else 1.0
    if str(page.get("lane") or "") in preferred_lanes:
        score += 1.5
    if str(page.get("address") or "") in visible_addresses or bool(page.get("model_visible")):
        score += 1.0
    if bool(page.get("in_active_snapshot")):
        score += 0.5
    if page.get("status") == "active":
        score += 0.25
    return score


def select_protected_pages(visible_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    protected = []
    for row in visible_pages:
        if not isinstance(row, dict):
            continue
        lane = str(row.get("lane") or "")
        execution = str(row.get("execution_class") or "")
        if lane in {"policy", "constraints_corrections", "task_state"} or execution in {"constitutional_policy", "authorized_task_state"}:
            protected.append(summarize_page(row, score=999.0))
    return protected[:6]


def summarize_page(page: dict[str, Any], *, score: float) -> dict[str, Any]:
    return {
        "address": str(page.get("address") or ""),
        "title": str(page.get("title") or ""),
        "lane": str(page.get("lane") or ""),
        "type": str(page.get("type") or ""),
        "execution_class": str(page.get("execution_class") or ""),
        "source_path": str(page.get("source_path") or ""),
        "model_visible": bool(page.get("model_visible", page.get("materialized_text") is not None)),
        "fault_count": int(page.get("fault_count") or 0),
        "taints": list_value(page.get("taints")),
        "score": round(float(score), 3),
    }


def dedupe_page_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        address = str(row.get("address") or "")
        if not address or address in seen:
            continue
        seen.add(address)
        out.append(row)
    return out


def build_guard_results(
    *,
    family: dict[str, Any],
    defaults: dict[str, Any],
    compiled: dict[str, Any],
    probe: dict[str, Any],
    status: dict[str, Any],
    training: dict[str, Any],
    runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    required_writebacks = list_value(family.get("required_writebacks")) or list_value(defaults.get("write_back_event_kinds"))
    return [
        guard("snapshot_present", not defaults.get("require_snapshot", True) or bool(compiled.get("snapshot")), compiled.get("snapshot"), "hard"),
        guard("probe_green_or_yellow", probe.get("trigger_state") in {"GREEN", "YELLOW"}, probe.get("trigger_state"), "hard"),
        guard("training_admission_green", training.get("trigger_state") == "GREEN", training.get("trigger_state"), "hard"),
        guard("public_training_rows_zero", public_training_rows(probe, status, training, runtime) == 0, public_training_rows(probe, status, training, runtime), "hard"),
        guard("external_inference_zero", external_calls(probe, status, training, runtime) == 0, external_calls(probe, status, training, runtime), "hard"),
        guard("fallback_returns_zero", fallback_returns(probe, status, training, runtime) == 0, fallback_returns(probe, status, training, runtime), "hard"),
        guard("writeback_events_declared", bool(required_writebacks), required_writebacks, "hard"),
        guard("runtime_not_claimed_by_bridge", not bool(get_path(runtime, ["summary", "runtime_profile_claimed"], False)), get_path(runtime, ["summary", "runtime_profile_claimed"], False), "hard"),
    ]


def system_gates(
    policy: dict[str, Any],
    probe: dict[str, Any],
    status: dict[str, Any],
    training: dict[str, Any],
    consumer: dict[str, Any],
    runtime: dict[str, Any],
    release: dict[str, Any],
    compiled: dict[str, Any],
    contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    high = [row for row in contexts if row.get("priority") == "high"]
    return [
        gate("policy_loaded", policy.get("policy") == "project_theseus_vcm_task_context_policy_v1", policy.get("policy"), "hard"),
        gate("compiled_context_present", bool(compiled.get("model_visible_pages")), len(list_value(compiled.get("model_visible_pages"))), "hard"),
        gate("snapshot_present", bool(compiled.get("snapshot")), compiled.get("snapshot"), "hard"),
        gate("vcm_probe_green_or_yellow", probe.get("trigger_state") in {"GREEN", "YELLOW"}, probe.get("trigger_state"), "hard"),
        gate("training_admission_green", training.get("trigger_state") == "GREEN", training.get("trigger_state"), "hard"),
        gate("consumer_audit_green", consumer.get("trigger_state") == "GREEN", consumer.get("trigger_state"), "warning"),
        gate(
            "release_conformance_core_ready",
            bool(get_path(release, ["summary", "core_profiles_ready"], False)),
            get_path(release, ["summary", "profile_states"], {}),
            "warning",
        ),
        gate("vcm_runtime_not_claimed", not bool(get_path(runtime, ["summary", "runtime_profile_claimed"], False)), get_path(runtime, ["summary", "runtime_profile_claimed"], False), "hard"),
        gate("high_priority_task_families_ready", all(row.get("ready") for row in high), [row.get("task_family_id") for row in high if not row.get("ready")], "hard"),
        gate("required_task_family_count", len(contexts) >= 8, len(contexts), "hard"),
    ]


def task_family_gates(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gates = []
    for row in contexts:
        gates.append(
            gate(
                f"task_context_{row['task_family_id']}_ready",
                bool(row.get("ready")),
                {
                    "selected_pages": len(row.get("selected_pages") or []),
                    "blockers": row.get("blockers") or [],
                    "priority": row.get("priority"),
                },
                "hard" if row.get("priority") == "high" else "warning",
            )
        )
    return gates


def guard(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"guard": name, "passed": bool(passed), "evidence": evidence, "severity": severity}


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence, "severity": severity}


def external_calls(*reports: dict[str, Any]) -> int:
    return sum(int(get_path(report, ["summary", "external_inference_calls"], report.get("external_inference_calls", 0)) or 0) for report in reports if isinstance(report, dict))


def public_training_rows(*reports: dict[str, Any]) -> int:
    total = 0
    for report in reports:
        if not isinstance(report, dict):
            continue
        total += int(get_path(report, ["summary", "public_training_rows_written"], report.get("public_training_rows_written", 0)) or 0)
        total += int(get_path(report, ["summary", "public_training_rows"], 0) or 0)
    return total


def fallback_returns(*reports: dict[str, Any]) -> int:
    return sum(int(get_path(report, ["summary", "fallback_return_count"], report.get("fallback_return_count", 0)) or 0) for report in reports if isinstance(report, dict))


def stable_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_value(report.get("summary"))
    lines = [
        "# VCM Task Context Bridge",
        "",
        f"- Trigger: `{report.get('trigger_state')}`",
        f"- Task families ready: `{summary.get('ready_task_family_count')}/{summary.get('task_family_count')}`",
        f"- High-priority ready: `{summary.get('high_priority_ready_count')}/{summary.get('high_priority_task_family_count')}`",
        f"- Unique selected pages: `{summary.get('unique_selected_page_count')}`",
        f"- Snapshot: `{summary.get('snapshot')}`",
        f"- VCM probe: `{summary.get('vcm_probe_state')}`",
        f"- Training admission: `{summary.get('training_admission_state')}`",
        f"- VCM-Runtime claimed: `{summary.get('runtime_profile_claimed')}`",
        f"- Public training / external / fallback: `{summary.get('public_training_rows_written')}` / `{summary.get('external_inference_calls')}` / `{summary.get('fallback_return_count')}`",
        "",
        "## Task Families",
        "",
    ]
    for row in report.get("task_contexts") or []:
        lines.append(
            f"- `{row.get('task_family_id')}`: ready `{row.get('ready')}`, priority `{row.get('priority')}`, "
            f"pages `{len(row.get('selected_pages') or [])}`, blockers `{row.get('blockers') or []}`"
        )
    lines.extend(
        [
            "",
            "This report is the task-facing VCM integration contract. It does not claim native KV-cache runtime integration.",
            "",
        ]
    )
    return "\n".join(lines)


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def get_path(value: Any, path: list[Any], default: Any = None) -> Any:
    cur = value
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
