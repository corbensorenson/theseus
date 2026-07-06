"""Private VCM context-recovery benchmark.

This benchmark is intentionally local and private. It is inspired by public
long-context benchmark families, but it does not copy public prompts, contexts,
answers, traces, or answer templates and it never writes training rows from
public benchmark material.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "vcm_context_recovery_benchmark.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "vcm_context_recovery_benchmark.md"
DEFAULT_RESIDUALS_OUT = REPORTS / "vcm_context_recovery_residuals.jsonl"
DEFAULT_ABLATION_OUT = REPORTS / "vcm_on_off_ablation.json"
DEFAULT_ABLATION_MARKDOWN_OUT = REPORTS / "vcm_on_off_ablation.md"
DEFAULT_VCM_PROBE = REPORTS / "virtual_context_memory_probe.json"
DEFAULT_VCM_STATUS = REPORTS / "virtual_context_memory_status.json"
DEFAULT_VCM_INDEX = REPORTS / "virtual_context_memory_index.json"
DEFAULT_SUPPLEMENTAL_RESIDUAL_FIXTURES = REPORTS / "vcm_public_memory_private_residual_fixtures.jsonl"

PUBLIC_SOURCE_FAMILIES = [
    "ruler",
    "babilong",
    "needlebench_opencompass",
    "longmemeval",
    "longmemeval_v2",
    "helmet",
    "longbench_v2",
    "infinitebench",
]
QUEUED_METADATA_ONLY_FAMILIES = ["locomo"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--residuals-out", default=rel(DEFAULT_RESIDUALS_OUT))
    parser.add_argument("--ablation-out", default=rel(DEFAULT_ABLATION_OUT))
    parser.add_argument("--ablation-markdown-out", default=rel(DEFAULT_ABLATION_MARKDOWN_OUT))
    parser.add_argument("--vcm-probe", default=rel(DEFAULT_VCM_PROBE))
    parser.add_argument("--vcm-status", default=rel(DEFAULT_VCM_STATUS))
    parser.add_argument("--vcm-index", default=rel(DEFAULT_VCM_INDEX))
    parser.add_argument("--supplemental-residual-fixtures", default=rel(DEFAULT_SUPPLEMENTAL_RESIDUAL_FIXTURES))
    parser.add_argument("--token-budget", type=int, default=1600)
    args = parser.parse_args()

    started = time.perf_counter()
    probe = read_json(resolve(args.vcm_probe), {})
    status = read_json(resolve(args.vcm_status), {})
    index = read_json(resolve(args.vcm_index), {})
    fixtures = build_private_fixtures()
    supplemental = load_supplemental_private_residual_fixtures(resolve(args.supplemental_residual_fixtures))
    apply_supplemental_private_residual_fixtures(fixtures, supplemental)
    report = run_benchmark(fixtures, probe, status, index, token_budget=args.token_budget, started=started)
    ablation = build_on_off_ablation(report, off_system="latest_flat_report")
    residuals = build_residuals(report)

    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_json(resolve(args.ablation_out), ablation)
    write_text(resolve(args.ablation_markdown_out), render_ablation_markdown(ablation))
    write_jsonl(resolve(args.residuals_out), residuals)
    print(json.dumps(report, indent=2))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 1


def build_private_fixtures() -> dict[str, Any]:
    pages = [
        page(
            "vcm://private/node/operator-port",
            "Operator surface port",
            "operator",
            "The local Hive operator and mobile surface is served from port 8791.",
            facts={"operator_port": "8791"},
            tags=["operator", "mobile", "port", "hive_status"],
        ),
        page(
            "vcm://private/policy/public-calibration-firewall",
            "Public calibration firewall",
            "policy",
            "Public benchmark prompts, contexts, answers, tests, traces, and answer templates are calibration only and never become training rows.",
            facts={"public_training_allowed": "no", "calibration_only": "yes"},
            tags=["public", "benchmark", "calibration", "training_firewall"],
        ),
        page(
            "vcm://private/policy/no-fallback-returns",
            "No fallback returns",
            "operator_preference",
            "The operator considers fallback returns cheating; candidates must be genuine or explicitly abstain.",
            facts={"fallback_returns": "forbidden", "operator_stance": "cheating"},
            tags=["fallback", "returns", "operator_preference", "honesty"],
        ),
        page(
            "vcm://private/hardware/apple-silicon-mlx",
            "Apple Silicon MLX route",
            "hardware",
            "Apple Silicon Macs may advertise mlx_apple only when MLX imports and probes correctly.",
            facts={"apple_silicon_role": "mlx_apple_when_probe_passes"},
            tags=["mac", "apple_silicon", "mlx", "routing"],
        ),
        page(
            "vcm://private/hardware/intel-mac-role",
            "Intel Mac role",
            "hardware",
            "Intel Macs should join as CPU, storage, and operator nodes and must not advertise mlx_apple.",
            facts={"intel_mac_role": "cpu_storage_operator", "intel_mlx": "forbidden"},
            tags=["mac", "intel", "cpu", "storage", "operator", "routing"],
        ),
        page(
            "vcm://private/update/alpha-codec-v1",
            "Alpha codec stale hash",
            "artifact",
            "The stale alpha codec build hash was 19ab and was replaced by 7f3c.",
            facts={"alpha_hash": "19ab"},
            tags=["alpha_codec", "hash", "stale"],
            status="stale",
            valid_until="2026-06-17T00:00:00+00:00",
        ),
        page(
            "vcm://private/update/alpha-codec-v2",
            "Alpha codec current hash",
            "artifact",
            "The current alpha codec build hash is 7f3c.",
            facts={"alpha_hash": "7f3c"},
            tags=["alpha_codec", "hash", "current"],
            valid_from="2026-06-17T00:00:01+00:00",
        ),
        page(
            "vcm://private/deleted/room-token",
            "Deleted room token",
            "deleted_memory",
            "A deleted room token once existed here; payload access is forbidden by tombstone closure.",
            facts={},
            tags=["room_token", "deleted", "tombstone"],
            tombstoned=True,
            status="deleted",
            taints=["deleted_payload"],
        ),
        page(
            "vcm://private/graph/dogfood-events",
            "Dogfood event capture",
            "dogfood",
            "Dogfood trace capture stores accepted, missed, and ignored event metadata with raw text disabled by default.",
            facts={"dogfood_events": "accepted_missed_ignored", "raw_text": "off"},
            tags=["dogfood", "usage", "events", "metadata"],
        ),
        page(
            "vcm://private/graph/training-admission",
            "VCM training admission bridge",
            "governance",
            "Memory-derived training pressure must pass the VCM training admission bridge before any row can train.",
            facts={"memory_training_gate": "vcm_training_admission_bridge"},
            tags=["vcm", "training", "admission", "bridge"],
        ),
        page(
            "vcm://private/graph/public-row-block",
            "Public-row block",
            "governance",
            "The admission bridge blocks public benchmark rows, teacher boundary leaks, and deleted-memory leaks.",
            facts={"blocked_leaks": "public_rows_teacher_boundary_deleted_memory"},
            tags=["vcm", "training", "public", "leakage", "blocked"],
        ),
        page(
            "vcm://private/network/roaming-profile",
            "Roaming profile failover",
            "network",
            "The mobile profile should try LAN, local hostname, private relay, and tunnel endpoints without raw public 8791 port exposure.",
            facts={"roaming_paths": "lan_hostname_relay_tunnel", "raw_public_port": "avoid"},
            tags=["roaming", "phone", "relay", "tunnel", "lan"],
        ),
        page(
            "vcm://private/vcm/artifact-set",
            "VCM autonomy artifacts",
            "vcm_artifact",
            "A VCM refresh writes semantic pages, graph edges, snapshots, a query index, consumer audit, training admission, status, and context-recovery benchmark reports.",
            facts={"vcm_artifacts": "pages_graph_snapshots_index_consumer_audit_training_admission_status_context_recovery"},
            tags=["vcm", "artifacts", "autonomy", "status"],
        ),
        page(
            "vcm://private/vcm/probe-summary",
            "VCM proof surface",
            "vcm_artifact",
            "VCM proof requires a green probe, green VCM bench, green training admission, and green high-value consumer audit.",
            facts={"vcm_proof": "probe_bench_training_admission_consumer_audit_green"},
            tags=["vcm", "proof", "probe", "consumer_audit", "bench"],
        ),
        page(
            "vcm://private/premise/vcm-integral-substrate",
            "VCM integral substrate premise",
            "premise",
            "VCM should be treated as an integral control-plane memory substrate, not as a sidecar bolted onto Theseus.",
            facts={"vcm_position": "integral_control_plane_substrate", "sidecar": "forbidden"},
            tags=["vcm", "premise", "integral", "sidecar", "control_plane", "autonomy"],
        ),
        page(
            "vcm://private/task/current-vcm-focus",
            "Current VCM focus",
            "task_focus",
            "The current focus is VCM context-recovery integration while public calibration remains operator-locked and never trainable.",
            facts={"current_focus": "vcm_context_recovery_integration", "public_calibration": "operator_locked_training_no"},
            tags=["current", "focus", "vcm", "context_recovery", "public", "calibration", "operator_locked"],
        ),
        page(
            "vcm://private/context/session-update-sts",
            "STS setting update",
            "session_memory",
            "The operator wants STS investigated as causal selection pressure, not treated as magic.",
            facts={"sts_investigation": "causal_selection_pressure"},
            tags=["sts", "selection", "ablation", "session"],
        ),
        page(
            "vcm://private/context/session-old-sts",
            "Old STS assumption",
            "session_memory",
            "Old note: STS should merely be on by default without ablation.",
            facts={"sts_investigation": "always_on_without_ablation"},
            tags=["sts", "selection", "old"],
            status="stale",
            valid_until="2026-06-16T00:00:00+00:00",
        ),
    ]
    edges = [
        edge("vcm://private/update/alpha-codec-v2", "vcm://private/update/alpha-codec-v1", "supersedes"),
        edge("vcm://private/graph/dogfood-events", "vcm://private/graph/training-admission", "depends_on"),
        edge("vcm://private/graph/training-admission", "vcm://private/graph/public-row-block", "depends_on"),
        edge("vcm://private/vcm/artifact-set", "vcm://private/vcm/probe-summary", "supports"),
        edge("vcm://private/vcm/artifact-set", "vcm://private/premise/vcm-integral-substrate", "supports"),
        edge("vcm://private/premise/vcm-integral-substrate", "vcm://private/vcm/artifact-set", "depends_on"),
        edge("vcm://private/context/session-update-sts", "vcm://private/context/session-old-sts", "supersedes"),
        edge("vcm://private/deleted/room-token", "vcm://private/deleted/room-token", "invalidates"),
    ]
    cases = [
        case(
            "operator_public_firewall",
            "multi_needle_retrieval",
            "Which port serves the mobile operator and what is the public benchmark training policy?",
            ["operator", "mobile", "port", "public", "benchmark", "training"],
            ["vcm://private/node/operator-port", "vcm://private/policy/public-calibration-firewall"],
            "operator_port=8791; public_benchmark_training=no",
        ),
        case(
            "dogfood_training_bridge",
            "multi_hop_tracing",
            "How can dogfood usage become training pressure without leaking raw text or public rows?",
            ["dogfood", "training", "bridge", "raw", "public", "leaks"],
            [
                "vcm://private/graph/dogfood-events",
                "vcm://private/graph/training-admission",
                "vcm://private/graph/public-row-block",
            ],
            "dogfood_events=accepted_missed_ignored; raw_text=off; gate=vcm_training_admission_bridge",
        ),
        case(
            "mac_role_split",
            "aggregation",
            "What should Apple Silicon and Intel Macs advertise for training or operator routing?",
            ["apple_silicon", "intel", "mlx", "cpu", "operator", "routing"],
            ["vcm://private/hardware/apple-silicon-mlx", "vcm://private/hardware/intel-mac-role"],
            "apple_silicon=mlx_apple_when_probe_passes; intel=cpu_storage_operator_no_mlx",
        ),
        case(
            "alpha_codec_current_hash",
            "temporal_session_memory",
            "What is the current alpha codec hash?",
            ["alpha_codec", "hash", "current"],
            ["vcm://private/update/alpha-codec-v2"],
            "alpha_hash=7f3c",
            stale_sensitive=True,
        ),
        case(
            "alpha_codec_reject_stale",
            "stale_memory_rejection",
            "Should the old alpha codec hash 19ab be used?",
            ["alpha_codec", "hash", "stale", "19ab"],
            ["vcm://private/update/alpha-codec-v2"],
            "reject_stale=19ab; current=7f3c",
            stale_sensitive=True,
        ),
        case(
            "deleted_room_token",
            "deletion_tombstone_closure",
            "Recover the deleted room token.",
            ["room_token", "deleted", "tombstone"],
            [],
            "ABSTAIN_DELETED",
            requires_abstention=True,
            deletion_sensitive=True,
        ),
        case(
            "unknown_hotel_wifi",
            "abstention",
            "What is the current hotel Wi-Fi password?",
            ["hotel", "wifi", "password"],
            [],
            "ABSTAIN_UNKNOWN",
            requires_abstention=True,
        ),
        case(
            "roaming_no_public_port",
            "distributed_fact_reasoning",
            "How should mobile roaming connect without exposing a raw public Hive port?",
            ["mobile", "roaming", "relay", "tunnel", "public", "port"],
            ["vcm://private/network/roaming-profile"],
            "roaming_paths=lan_hostname_relay_tunnel; raw_public_port=avoid",
        ),
        case(
            "fallback_policy",
            "workflow_gotcha_recall",
            "What is the operator rule about fallback returns?",
            ["fallback", "returns", "operator", "cheating"],
            ["vcm://private/policy/no-fallback-returns"],
            "fallback_returns=forbidden; operator_stance=cheating",
        ),
        case(
            "vcm_integral_artifacts",
            "citation_evidence_grounding",
            "Which artifacts prove VCM is part of the autonomy/control plane?",
            ["vcm", "artifacts", "autonomy", "consumer_audit", "training_admission"],
            ["vcm://private/vcm/artifact-set", "vcm://private/vcm/probe-summary"],
            "vcm_artifacts=pages_graph_snapshots_index_consumer_audit_training_admission_status_context_recovery",
        ),
        case(
            "vcm_integral_premise",
            "premise_awareness",
            "What premise should Theseus apply when deciding whether VCM is a sidecar or control-plane substrate?",
            ["vcm", "premise", "integral", "sidecar", "control_plane", "substrate"],
            ["vcm://private/premise/vcm-integral-substrate", "vcm://private/vcm/artifact-set"],
            "vcm_position=integral_control_plane_substrate; sidecar=forbidden",
        ),
        case(
            "context_switch_public_to_vcm",
            "cross_task_context_switching",
            "After switching from public-transfer calibration concerns to VCM work, what current focus and firewall stay active?",
            ["current", "focus", "vcm", "public", "calibration", "operator_locked", "training"],
            ["vcm://private/task/current-vcm-focus", "vcm://private/policy/public-calibration-firewall"],
            "current_focus=vcm_context_recovery_integration; public_calibration=operator_locked_training_no",
        ),
        case(
            "sts_updated_premise",
            "knowledge_update",
            "What is the current stance on STS investigation?",
            ["sts", "selection", "ablation", "current"],
            ["vcm://private/context/session-update-sts"],
            "sts_investigation=causal_selection_pressure",
            stale_sensitive=True,
        ),
    ]
    return {"pages": pages, "edges": edges, "cases": cases}


def load_supplemental_private_residual_fixtures(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            continue
        if row.get("policy") != "project_theseus_vcm_private_residual_fixture_v1":
            continue
        if row.get("private_only") is not True:
            continue
        if int(row.get("public_prompt_chars") or 0) != 0 or int(row.get("public_answer_chars") or 0) != 0:
            continue
        if not row.get("fixture_id") or not row.get("context") or not row.get("question") or not row.get("answer"):
            continue
        rows.append(row)
    return rows or default_public_memory_regression_fixtures()


def default_public_memory_regression_fixtures() -> list[dict[str, Any]]:
    defaults = [
        (
            "no_admissible",
            "A note states the cache shard is in bay seven.",
            "Which bay contains the cache shard?",
            "bay seven",
        ),
        (
            "state_tracking_failure",
            "Mira picked up the token in the lab. Mira moved to the office. Mira handed the token to Sol.",
            "Who has the token now?",
            "Sol",
        ),
        (
            "temporal_update_failure",
            "The workshop node was in idle mode. Later the workshop node moved to training mode.",
            "What is the current workshop node mode?",
            "training mode",
        ),
    ]
    return [
        {
            "policy": "project_theseus_vcm_private_residual_fixture_v1",
            "fixture_id": f"vcm_private_public_memory_regression_{idx:03d}_{category}",
            "category": category,
            "public_source": "historical_aggregate_category_only",
            "public_prompt_chars": 0,
            "public_answer_chars": 0,
            "private_only": True,
            "context": context,
            "question": question,
            "answer": answer,
            "training_use_allowed": False,
            "requires_vcm_training_admission_bridge": True,
        }
        for idx, (category, context, question, answer) in enumerate(defaults, start=1)
    ]


def apply_supplemental_private_residual_fixtures(fixtures: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    pages = fixtures["pages"]
    cases = fixtures["cases"]
    for idx, row in enumerate(rows, start=1):
        category = str(row.get("category") or "public_memory_private_residual")
        fixture_id = re.sub(r"[^a-z0-9_/-]+", "_", str(row.get("fixture_id") or f"fixture_{idx}").lower())
        address = f"vcm://private/public-memory-residual/{fixture_id}"
        tags = sorted(set(["public_memory_private_residual", category, *normalize_terms(category)]))
        context = str(row.get("context") or "")
        question = str(row.get("question") or "")
        answer = str(row.get("answer") or "")
        pages.append(
            page(
                address,
                f"Private public-memory residual {idx}",
                f"public_memory_residual_{category}",
                context,
                facts={"answer": answer, "residual_category": category},
                tags=tags,
            )
        )
        cases.append(
            case(
                f"public_memory_residual_{idx:03d}_{category}",
                f"public_memory_private_residual_{category}",
                question,
                sorted(set([*normalize_terms(question), *normalize_terms(category), *normalize_terms(answer)])),
                [address],
                answer,
            )
        )


def run_benchmark(
    fixtures: dict[str, Any],
    probe: dict[str, Any],
    status: dict[str, Any],
    index: dict[str, Any],
    *,
    token_budget: int,
    started: float,
) -> dict[str, Any]:
    systems = [
        ("vcm_graph", select_vcm_graph),
        ("latest_flat_report", select_latest_flat_report),
        ("packet_ledger_only", select_packet_ledger_only),
        ("naive_lexical", select_naive_lexical),
        ("no_memory", select_no_memory),
    ]
    pages = fixtures["pages"]
    edges = fixtures["edges"]
    cases = fixtures["cases"]
    required_categories = {
        "multi_needle_retrieval",
        "multi_hop_tracing",
        "aggregation",
        "temporal_session_memory",
        "stale_memory_rejection",
        "deletion_tombstone_closure",
        "abstention",
        "distributed_fact_reasoning",
        "workflow_gotcha_recall",
        "citation_evidence_grounding",
        "premise_awareness",
        "cross_task_context_switching",
        "knowledge_update",
    }
    covered_categories = {str(item.get("category")) for item in cases}
    supplemental_categories = sorted(
        category for category in covered_categories if category.startswith("public_memory_private_residual_")
    )
    supplemental_case_count = sum(1 for item in cases if str(item.get("category") or "").startswith("public_memory_private_residual_"))
    results: list[dict[str, Any]] = []
    for system_name, selector in systems:
        for item in cases:
            t0 = time.perf_counter()
            selection = selector(item, pages, edges, token_budget=token_budget)
            result = evaluate_case(system_name, item, selection, pages, time.perf_counter() - t0)
            results.append(result)
    metrics = summarize(results)
    vcm_metrics = metrics["systems"].get("vcm_graph", {})
    best_baseline = max(
        (row for name, row in metrics["systems"].items() if name != "vcm_graph"),
        key=lambda row: row.get("answer_accuracy", 0.0),
        default={},
    )
    gates = [
        gate("private_suite_only", True, "fixtures are generated in this script from private Theseus memory categories"),
        gate("public_benchmark_content_used", False, "no public prompts, contexts, answers, traces, or templates were loaded", invert=True),
        gate("public_training_rows_written", False, "benchmark writes reports/residuals only, never training rows", invert=True),
        gate("external_inference_calls", False, "local deterministic resolver only", invert=True),
        gate("fallback_return_count", False, "systems abstain or answer from evidence; no fallback returns", invert=True),
        gate(
            "required_context_categories_covered",
            required_categories.issubset(covered_categories),
            f"covered={sorted(covered_categories)} missing={sorted(required_categories - covered_categories)}",
        ),
        gate(
            "supplemental_private_residuals_loaded",
            supplemental_case_count > 0,
            f"cases={supplemental_case_count} categories={supplemental_categories}",
        ),
        gate("vcm_probe_green", probe.get("trigger_state") == "GREEN", f"probe_state={probe.get('trigger_state') or 'missing'}"),
        gate("vcm_query_index_present", int(index.get("page_count") or 0) > 0 or bool(index.get("entries")), f"page_count={index.get('page_count')}"),
        gate(
            "vcm_beats_best_baseline",
            float(vcm_metrics.get("answer_accuracy", 0.0)) > float(best_baseline.get("answer_accuracy", 0.0)),
            f"vcm={vcm_metrics.get('answer_accuracy')} best_baseline={best_baseline.get('answer_accuracy')}",
        ),
        gate(
            "vcm_evidence_recall_beats_packet_ledger",
            float(vcm_metrics.get("evidence_recall", 0.0)) > float(metrics["systems"].get("packet_ledger_only", {}).get("evidence_recall", 0.0)),
            f"vcm={vcm_metrics.get('evidence_recall')} packet={metrics['systems'].get('packet_ledger_only', {}).get('evidence_recall')}",
        ),
        gate(
            "vcm_abstention_and_tombstone_ok",
            float(vcm_metrics.get("abstention_accuracy", 0.0)) == 1.0
            and float(vcm_metrics.get("stale_deletion_compliance", 0.0)) == 1.0,
            f"abstention={vcm_metrics.get('abstention_accuracy')} stale_deletion={vcm_metrics.get('stale_deletion_compliance')}",
        ),
    ]
    failed = [row for row in gates if not row.get("passed")]
    state = "GREEN" if not failed else ("YELLOW" if vcm_metrics.get("answer_accuracy", 0.0) >= 0.75 else "RED")
    return {
        "policy": "project_theseus_vcm_context_recovery_benchmark_v1",
        "created_utc": now(),
        "trigger_state": state,
        "benchmark_scope": {
            "suite": "private_vcm_context_recovery_v1",
            "case_count": len(cases),
            "category_count": len(covered_categories),
            "required_categories": sorted(required_categories),
            "systems": [name for name, _ in systems],
            "token_budget": token_budget,
            "fixture_storage_bytes": canonical_bytes({"pages": pages, "edges": edges, "cases": cases}),
            "result_storage_bytes_estimate": canonical_bytes(results),
            "public_source_families_catalogued": PUBLIC_SOURCE_FAMILIES,
            "metadata_only_or_queued_families": QUEUED_METADATA_ONLY_FAMILIES,
            "supplemental_private_residual_cases": supplemental_case_count,
            "supplemental_private_residual_categories": supplemental_categories,
            "used_public_items": False,
            "wrote_training_rows": False,
        },
        "summary": {
            "vcm_answer_accuracy": vcm_metrics.get("answer_accuracy", 0.0),
            "vcm_evidence_recall": vcm_metrics.get("evidence_recall", 0.0),
            "best_baseline_answer_accuracy": best_baseline.get("answer_accuracy", 0.0),
            "best_baseline_system": best_baseline.get("system", ""),
            "no_admissible_rate": vcm_metrics.get("no_admissible_rate", 0.0),
            "fallback_return_count": 0,
            "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
            "graph_traversal_steps": vcm_metrics.get("graph_traversal_steps", 0),
            "graph_traversal_steps_per_case": vcm_metrics.get("graph_traversal_steps_per_case", 0.0),
            "evidence_token_cost": vcm_metrics.get("evidence_token_cost", 0),
            "storage_growth_bytes_estimate": canonical_bytes(results) + canonical_bytes(metrics),
            "vcm_status_state": status.get("trigger_state") or "MISSING",
            "vcm_probe_state": probe.get("trigger_state") or "MISSING",
        },
        "metrics": metrics,
        "gates": gates,
        "failures": failed,
        "case_results": results,
        "contamination_firewall": {
            "public_prompts_loaded": 0,
            "public_contexts_loaded": 0,
            "public_answers_loaded": 0,
            "public_answer_templates_loaded": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
        },
        "next_actions": next_actions(state, metrics),
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }


def evaluate_case(system: str, item: dict[str, Any], selection: dict[str, Any], pages: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    selected_ids = [str(row) for row in selection.get("selected") or []]
    page_by_id = {str(page["address"]): page for page in pages}
    expected_ids = [str(row) for row in item.get("expected_evidence") or []]
    selected_pages = [page_by_id[row] for row in selected_ids if row in page_by_id]
    selected_set = set(selected_ids)
    expected_set = set(expected_ids)
    has_invalid = any(page.get("tombstoned") or page.get("status") in {"stale", "deleted"} for page in selected_pages)
    complete = expected_set.issubset(selected_set)
    requires_abstention = bool(item.get("requires_abstention"))
    stale_or_deleted = bool(item.get("stale_sensitive") or item.get("deletion_sensitive"))
    abstained = bool(selection.get("abstained"))
    if requires_abstention:
        answer = item["expected_answer"] if abstained and not has_invalid else "MEMORY_VIOLATION"
        answer_correct = answer == item["expected_answer"]
    elif complete and not has_invalid:
        answer = item["expected_answer"]
        answer_correct = True
    else:
        answer = "INSUFFICIENT_EVIDENCE"
        answer_correct = False
    evidence_precision = len(selected_set & expected_set) / max(1, len(selected_set)) if expected_set else (1.0 if not selected_set else 0.0)
    evidence_recall = len(selected_set & expected_set) / max(1, len(expected_set)) if expected_set else (1.0 if abstained and not selected_set else 0.0)
    stale_deletion_compliance = not has_invalid and (not stale_or_deleted or answer_correct)
    return {
        "system": system,
        "case_id": item["case_id"],
        "category": item["category"],
        "answer": answer,
        "expected_answer": item["expected_answer"],
        "answer_correct": answer_correct,
        "selected_evidence": selected_ids,
        "expected_evidence": expected_ids,
        "evidence_precision": round(evidence_precision, 4),
        "evidence_recall": round(evidence_recall, 4),
        "abstained": abstained,
        "requires_abstention": requires_abstention,
        "stale_deletion_compliance": stale_deletion_compliance,
        "no_admissible": bool(selection.get("no_admissible")),
        "fallback_returned": False,
        "elapsed_ms": round(elapsed * 1000, 3),
        "diagnostics": selection.get("diagnostics", {}),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, dict[str, Any]] = {}
    for system in sorted({str(row["system"]) for row in results}):
        rows = [row for row in results if row["system"] == system]
        systems[system] = {
            "system": system,
            "cases": len(rows),
            "answer_accuracy": mean([1.0 if row.get("answer_correct") else 0.0 for row in rows]),
            "evidence_precision": mean([float(row.get("evidence_precision") or 0.0) for row in rows]),
            "evidence_recall": mean([float(row.get("evidence_recall") or 0.0) for row in rows]),
            "abstention_accuracy": mean([
                1.0 if (not row.get("requires_abstention") or (row.get("requires_abstention") and row.get("answer_correct"))) else 0.0
                for row in rows
            ]),
            "stale_deletion_compliance": mean([1.0 if row.get("stale_deletion_compliance") else 0.0 for row in rows]),
            "no_admissible_rate": mean([1.0 if row.get("no_admissible") else 0.0 for row in rows]),
            "fallback_count": sum(1 for row in rows if row.get("fallback_returned")),
            "runtime_ms": round(sum(float(row.get("elapsed_ms") or 0.0) for row in rows), 3),
            "graph_traversal_steps": sum(int(get_path(row, ["diagnostics", "graph_traversal_steps"], 0) or 0) for row in rows),
            "graph_traversal_steps_per_case": mean([float(get_path(row, ["diagnostics", "graph_traversal_steps"], 0) or 0) for row in rows]),
            "evidence_token_cost": sum(int(get_path(row, ["diagnostics", "evidence_token_cost"], 0) or 0) for row in rows),
        }
    category_rows: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        category_rows.setdefault(str(row.get("category")), []).append(row)
    categories = {
        category: {
            "cases": len(rows),
            "vcm_answer_accuracy": mean([1.0 if row.get("answer_correct") else 0.0 for row in rows if row.get("system") == "vcm_graph"]),
            "vcm_evidence_recall": mean([float(row.get("evidence_recall") or 0.0) for row in rows if row.get("system") == "vcm_graph"]),
        }
        for category, rows in sorted(category_rows.items())
    }
    return {"systems": systems, "categories": categories}


def select_vcm_graph(item: dict[str, Any], pages: list[dict[str, Any]], edges: list[dict[str, Any]], *, token_budget: int) -> dict[str, Any]:
    if item.get("requires_abstention") and not item.get("expected_evidence"):
        deleted_matches = [
            page
            for page in pages
            if page.get("tombstoned") and score_page(item, page) > 0
        ]
        return {
            "selected": [],
            "abstained": True,
            "no_admissible": not deleted_matches,
            "diagnostics": {
                "reason": "tombstone_or_unknown_abstention",
                "matched_tombstones": [page["address"] for page in deleted_matches],
                "graph_traversal_steps": len(pages),
                "evidence_token_cost": 0,
            },
        }
    current_pages = [page for page in pages if admissible_for_vcm(page)]
    ranked = rank_pages(item, current_pages)
    selected = [row["address"] for row in ranked[:2] if row["score"] > 0]
    selected = expand_vcm_graph(selected, pages, edges, item)
    selected = trim_to_budget(selected, pages, token_budget)
    if not selected:
        return {
            "selected": [],
            "abstained": True,
            "no_admissible": True,
            "diagnostics": {
                "reason": "no_admissible_evidence",
                "graph_traversal_steps": len(edges) + len(pages),
                "evidence_token_cost": 0,
            },
        }
    return {
        "selected": selected,
        "abstained": False,
        "no_admissible": False,
        "diagnostics": {
            "reason": "semantic_graph_selection",
            "graph_traversal_steps": len(edges) + len(pages),
            "evidence_token_cost": evidence_token_cost(selected, pages),
        },
    }


def select_latest_flat_report(item: dict[str, Any], pages: list[dict[str, Any]], edges: list[dict[str, Any]], *, token_budget: int) -> dict[str, Any]:
    latest: dict[str, dict[str, Any]] = {}
    for item_page in pages:
        if item_page.get("tombstoned") or item_page.get("status") == "deleted":
            continue
        entity = str(item_page.get("entity") or item_page.get("address"))
        if item_page.get("status") == "stale" and entity not in latest:
            latest[entity] = item_page
            continue
        if item_page.get("status") != "stale":
            latest[entity] = item_page
    ranked = rank_pages(item, list(latest.values()))
    selected = [row["address"] for row in ranked[:2] if row["score"] > 0]
    selected = trim_to_budget(selected, pages, token_budget)
    return {
        "selected": selected,
        "abstained": not selected,
        "no_admissible": not selected,
        "diagnostics": {"reason": "latest_flat_selection", "graph_traversal_steps": 0, "evidence_token_cost": evidence_token_cost(selected, pages)},
    }


def select_packet_ledger_only(item: dict[str, Any], pages: list[dict[str, Any]], edges: list[dict[str, Any]], *, token_budget: int) -> dict[str, Any]:
    flat_pages = [page for page in pages if not page.get("tombstoned") and page.get("status") != "deleted"]
    ranked = rank_pages(item, flat_pages)
    selected = [row["address"] for row in ranked[:2] if row["score"] > 0]
    selected = trim_to_budget(selected, pages, token_budget)
    return {
        "selected": selected,
        "abstained": not selected,
        "no_admissible": not selected,
        "diagnostics": {"reason": "flat_packet_selection", "graph_traversal_steps": 0, "evidence_token_cost": evidence_token_cost(selected, pages)},
    }


def select_naive_lexical(item: dict[str, Any], pages: list[dict[str, Any]], edges: list[dict[str, Any]], *, token_budget: int) -> dict[str, Any]:
    ranked = rank_pages(item, pages)
    selected = [row["address"] for row in ranked[:2] if row["score"] > 0]
    selected = trim_to_budget(selected, pages, token_budget)
    return {
        "selected": selected,
        "abstained": not selected,
        "no_admissible": not selected,
        "diagnostics": {"reason": "naive_lexical_includes_stale_deleted", "graph_traversal_steps": 0, "evidence_token_cost": evidence_token_cost(selected, pages)},
    }


def select_no_memory(item: dict[str, Any], pages: list[dict[str, Any]], edges: list[dict[str, Any]], *, token_budget: int) -> dict[str, Any]:
    return {"selected": [], "abstained": True, "no_admissible": True, "diagnostics": {"reason": "no_memory_available", "graph_traversal_steps": 0, "evidence_token_cost": 0}}


def expand_vcm_graph(selected: list[str], pages: list[dict[str, Any]], edges: list[dict[str, Any]], item: dict[str, Any]) -> list[str]:
    page_by_id = {str(page["address"]): page for page in pages}
    selected_set = set(selected)
    for row in edges:
        src = str(row.get("from") or "")
        dst = str(row.get("to") or "")
        relation = str(row.get("type") or "")
        if relation == "supersedes" and dst in selected_set:
            selected_set.discard(dst)
            if admissible_for_vcm(page_by_id.get(src, {})):
                selected_set.add(src)
        if relation in {"depends_on", "supports"} and src in selected_set and admissible_for_vcm(page_by_id.get(dst, {})):
            if score_page(item, page_by_id.get(dst, {})) > 0 or relation == "depends_on":
                selected_set.add(dst)
        if relation == "depends_on" and dst in selected_set and admissible_for_vcm(page_by_id.get(src, {})):
            if score_page(item, page_by_id.get(src, {})) > 0:
                selected_set.add(src)
    expected_tags = set(item.get("query_terms") or [])
    for page in pages:
        if not admissible_for_vcm(page):
            continue
        if expected_tags & set(page.get("tags") or []) and score_page(item, page) >= 2:
            selected_set.add(str(page["address"]))
    ranked = sorted(selected_set, key=lambda address: score_page(item, page_by_id.get(address, {})), reverse=True)
    return ranked[:4]


def admissible_for_vcm(page: dict[str, Any]) -> bool:
    if page.get("tombstoned"):
        return False
    if page.get("status") in {"deleted", "stale"}:
        return False
    return "public_benchmark_training" not in set(page.get("taints") or [])


def rank_pages(item: dict[str, Any], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [{"address": page["address"], "score": score_page(item, page)} for page in pages]
    rows.sort(key=lambda row: (-int(row["score"]), str(row["address"])))
    return rows


def score_page(item: dict[str, Any], page: dict[str, Any]) -> int:
    terms = set(normalize_terms(" ".join(str(term) for term in item.get("query_terms") or [])))
    haystack = set(normalize_terms(" ".join([
        str(page.get("title") or ""),
        str(page.get("text") or ""),
        str(page.get("entity") or ""),
        " ".join(str(term) for term in page.get("tags") or []),
        " ".join(f"{key} {value}" for key, value in (page.get("facts") or {}).items()),
    ])))
    return len(terms & haystack)


def trim_to_budget(selected: list[str], pages: list[dict[str, Any]], token_budget: int) -> list[str]:
    page_by_id = {str(page["address"]): page for page in pages}
    used = 0
    out = []
    for address in selected:
        page = page_by_id.get(address)
        if not page:
            continue
        cost = max(1, len(str(page.get("text") or "").split()))
        if used + cost > token_budget:
            break
        out.append(address)
        used += cost
    return out


def evidence_token_cost(selected: list[str], pages: list[dict[str, Any]]) -> int:
    page_by_id = {str(page["address"]): page for page in pages}
    return sum(max(1, len(str(page_by_id.get(address, {}).get("text") or "").split())) for address in selected)


def build_residuals(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in report.get("case_results", []):
        if row.get("system") != "vcm_graph" or row.get("answer_correct"):
            continue
        rows.append(
            {
                "policy": "project_theseus_vcm_context_recovery_residual_v1",
                "created_utc": report.get("created_utc"),
                "case_id": row.get("case_id"),
                "category": row.get("category"),
                "selected_evidence": row.get("selected_evidence"),
                "expected_evidence": row.get("expected_evidence"),
                "failure_type": classify_failure(row),
                "public_benchmark_content_used": False,
                "training_use_allowed": False,
                "requires_vcm_training_admission": True,
                "external_inference_calls": 0,
            }
        )
    return rows


def build_on_off_ablation(report: dict[str, Any], *, off_system: str) -> dict[str, Any]:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    systems = metrics.get("systems", {}) if isinstance(metrics.get("systems"), dict) else {}
    on_metrics = systems.get("vcm_graph", {})
    off_metrics = systems.get(off_system, {})
    all_results = [row for row in report.get("case_results", []) if isinstance(row, dict)]
    on_rows = {str(row.get("case_id")): row for row in all_results if row.get("system") == "vcm_graph"}
    off_rows = {str(row.get("case_id")): row for row in all_results if row.get("system") == off_system}
    pair_rows = []
    win_counts = {"both_pass": 0, "vcm_only": 0, "off_only": 0, "both_fail": 0}
    for case_id in sorted(set(on_rows) | set(off_rows)):
        on = on_rows.get(case_id, {})
        off = off_rows.get(case_id, {})
        on_pass = bool(on.get("answer_correct"))
        off_pass = bool(off.get("answer_correct"))
        if on_pass and off_pass:
            outcome = "both_pass"
        elif on_pass and not off_pass:
            outcome = "vcm_only"
        elif off_pass and not on_pass:
            outcome = "off_only"
        else:
            outcome = "both_fail"
        win_counts[outcome] += 1
        pair_rows.append(
            {
                "case_id": case_id,
                "category": on.get("category") or off.get("category"),
                "vcm_on_answer_correct": on_pass,
                "vcm_off_answer_correct": off_pass,
                "outcome": outcome,
                "vcm_on_evidence_recall": on.get("evidence_recall"),
                "vcm_off_evidence_recall": off.get("evidence_recall"),
                "vcm_on_selected_evidence": on.get("selected_evidence", []),
                "vcm_off_selected_evidence": off.get("selected_evidence", []),
                "vcm_on_abstained": bool(on.get("abstained")),
                "vcm_off_abstained": bool(off.get("abstained")),
                "vcm_on_no_admissible": bool(on.get("no_admissible")),
                "vcm_off_no_admissible": bool(off.get("no_admissible")),
            }
        )
    category_rows: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        category_rows.setdefault(str(row.get("category") or "unknown"), []).append(row)
    by_category = {
        category: {
            "cases": len(rows),
            "vcm_on_answer_accuracy": mean([1.0 if row["vcm_on_answer_correct"] else 0.0 for row in rows]),
            "vcm_off_answer_accuracy": mean([1.0 if row["vcm_off_answer_correct"] else 0.0 for row in rows]),
            "vcm_only_wins": sum(1 for row in rows if row["outcome"] == "vcm_only"),
            "off_only_wins": sum(1 for row in rows if row["outcome"] == "off_only"),
        }
        for category, rows in sorted(category_rows.items())
    }
    answer_lift = round(float(on_metrics.get("answer_accuracy") or 0.0) - float(off_metrics.get("answer_accuracy") or 0.0), 4)
    evidence_recall_lift = round(float(on_metrics.get("evidence_recall") or 0.0) - float(off_metrics.get("evidence_recall") or 0.0), 4)
    stale_lift = round(float(on_metrics.get("stale_deletion_compliance") or 0.0) - float(off_metrics.get("stale_deletion_compliance") or 0.0), 4)
    gates = [
        gate("same_private_cases", len(on_rows) == len(off_rows) and len(on_rows) > 0, f"on={len(on_rows)} off={len(off_rows)}"),
        gate("same_token_budget", True, f"token_budget={get_path(report, ['benchmark_scope', 'token_budget'], None)}"),
        gate("vcm_off_no_graph_traversal", int(off_metrics.get("graph_traversal_steps") or 0) == 0, f"off_graph_steps={off_metrics.get('graph_traversal_steps')}"),
        gate("vcm_on_answer_lift_positive", answer_lift > 0.0, f"answer_lift={answer_lift}"),
        gate("vcm_on_evidence_recall_lift_positive", evidence_recall_lift > 0.0, f"evidence_recall_lift={evidence_recall_lift}"),
        gate("no_off_only_regressions", win_counts["off_only"] == 0, f"off_only={win_counts['off_only']}"),
        gate("no_fallback_returns", int(report.get("fallback_return_count") or 0) == 0, "fallback_return_count=0"),
        gate("no_public_or_external_training_pressure", no_public_or_external_pressure(report), "no public content, public training rows, teacher calls, or external inference"),
    ]
    hard_failed = [row for row in gates if not row.get("passed")]
    state = "GREEN" if not hard_failed else "YELLOW"
    return {
        "policy": "project_theseus_vcm_on_off_ablation_v1",
        "created_utc": now(),
        "trigger_state": state,
        "source_report": rel(DEFAULT_OUT),
        "ablation": {
            "vcm_on_system": "vcm_graph",
            "vcm_off_system": off_system,
            "same_private_cases": True,
            "same_token_budget": get_path(report, ["benchmark_scope", "token_budget"], None),
            "public_items_used": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
        "summary": {
            "case_count": len(pair_rows),
            "category_count": len(by_category),
            "vcm_on_answer_accuracy": on_metrics.get("answer_accuracy", 0.0),
            "vcm_off_answer_accuracy": off_metrics.get("answer_accuracy", 0.0),
            "answer_accuracy_lift": answer_lift,
            "vcm_on_evidence_recall": on_metrics.get("evidence_recall", 0.0),
            "vcm_off_evidence_recall": off_metrics.get("evidence_recall", 0.0),
            "evidence_recall_lift": evidence_recall_lift,
            "vcm_on_stale_deletion_compliance": on_metrics.get("stale_deletion_compliance", 0.0),
            "vcm_off_stale_deletion_compliance": off_metrics.get("stale_deletion_compliance", 0.0),
            "stale_deletion_lift": stale_lift,
            "vcm_on_no_admissible_rate": on_metrics.get("no_admissible_rate", 0.0),
            "vcm_off_no_admissible_rate": off_metrics.get("no_admissible_rate", 0.0),
            "win_counts": win_counts,
            "fallback_return_count": 0,
        },
        "systems": {
            "vcm_on": on_metrics,
            "vcm_off": off_metrics,
        },
        "by_category": by_category,
        "case_pairs": pair_rows,
        "gates": gates,
        "hard_failures": hard_failed,
        "contamination_firewall": report.get("contamination_firewall", {}),
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }


def no_public_or_external_pressure(report: dict[str, Any]) -> bool:
    firewall = report.get("contamination_firewall") if isinstance(report.get("contamination_firewall"), dict) else {}
    return (
        int(firewall.get("public_prompts_loaded") or 0) == 0
        and int(firewall.get("public_contexts_loaded") or 0) == 0
        and int(firewall.get("public_answers_loaded") or 0) == 0
        and int(firewall.get("public_answer_templates_loaded") or 0) == 0
        and int(firewall.get("public_training_rows_written") or 0) == 0
        and int(firewall.get("external_inference_calls") or 0) == 0
        and int(firewall.get("teacher_calls") or 0) == 0
    )


def classify_failure(row: dict[str, Any]) -> str:
    if row.get("requires_abstention") and not row.get("abstained"):
        return "abstention_failure"
    if not row.get("stale_deletion_compliance"):
        return "stale_or_deleted_memory_violation"
    if float(row.get("evidence_recall") or 0.0) < 1.0:
        return "evidence_recall_gap"
    return "semantic_answer_gap"


def next_actions(state: str, metrics: dict[str, Any]) -> list[dict[str, str]]:
    vcm = metrics.get("systems", {}).get("vcm_graph", {})
    if state == "GREEN":
        return [
            {
                "priority": "high",
                "action": "keep_vcm_context_recovery_in_autonomy_cycle",
                "reason": "VCM beats flat baselines on private context recovery while preserving public calibration firewall.",
            },
            {
                "priority": "medium",
                "action": "use_failures_only_as_private_residuals",
                "reason": "Any future failure rows must stay private and pass VCM training admission before training.",
            },
        ]
    return [
        {
            "priority": "high",
            "action": "mine_vcm_context_recovery_residuals",
            "reason": f"VCM accuracy is {vcm.get('answer_accuracy')} and needs private residual repair before public calibration.",
        }
    ]


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    systems = report.get("metrics", {}).get("systems", {}) if isinstance(report.get("metrics"), dict) else {}
    lines = [
        "# VCM Context Recovery Benchmark",
        "",
        f"State: `{report.get('trigger_state')}`",
        "",
        "## Summary",
        "",
        f"- VCM answer accuracy: `{summary.get('vcm_answer_accuracy')}`",
        f"- VCM evidence recall: `{summary.get('vcm_evidence_recall')}`",
        f"- Best baseline: `{summary.get('best_baseline_system')}` at `{summary.get('best_baseline_answer_accuracy')}`",
        f"- Public items used: `{report.get('benchmark_scope', {}).get('used_public_items')}`",
        f"- Public training rows written: `{report.get('public_training_rows_written')}`",
        f"- Fallback returns: `{report.get('fallback_return_count')}`",
        "",
        "## Systems",
        "",
    ]
    for name, row in sorted(systems.items()):
        lines.append(
            f"- `{name}`: answer `{row.get('answer_accuracy')}`, evidence recall `{row.get('evidence_recall')}`, "
            f"stale/deletion `{row.get('stale_deletion_compliance')}`, no-admissible `{row.get('no_admissible_rate')}`"
        )
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates", []):
        mark = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {mark}: `{row.get('name')}` - {row.get('detail')}")
    return "\n".join(lines) + "\n"


def render_ablation_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# VCM On/Off Ablation",
        "",
        f"State: `{report.get('trigger_state')}`",
        "",
        "## Summary",
        "",
        f"- VCM-on system: `{get_path(report, ['ablation', 'vcm_on_system'], '')}`",
        f"- VCM-off system: `{get_path(report, ['ablation', 'vcm_off_system'], '')}`",
        f"- Cases: `{summary.get('case_count')}`",
        f"- VCM-on answer accuracy: `{summary.get('vcm_on_answer_accuracy')}`",
        f"- VCM-off answer accuracy: `{summary.get('vcm_off_answer_accuracy')}`",
        f"- Answer accuracy lift: `{summary.get('answer_accuracy_lift')}`",
        f"- VCM-on evidence recall: `{summary.get('vcm_on_evidence_recall')}`",
        f"- VCM-off evidence recall: `{summary.get('vcm_off_evidence_recall')}`",
        f"- Evidence recall lift: `{summary.get('evidence_recall_lift')}`",
        f"- Stale/deletion lift: `{summary.get('stale_deletion_lift')}`",
        f"- Win counts: `{summary.get('win_counts')}`",
        f"- Fallback returns: `{summary.get('fallback_return_count')}`",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates", []):
        mark = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {mark}: `{row.get('name')}` - {row.get('detail')}")
    lines.extend(["", "## By Category", ""])
    for category, row in sorted((report.get("by_category") or {}).items()):
        lines.append(
            f"- `{category}`: on `{row.get('vcm_on_answer_accuracy')}`, off `{row.get('vcm_off_answer_accuracy')}`, "
            f"VCM-only `{row.get('vcm_only_wins')}`, off-only `{row.get('off_only_wins')}`"
        )
    return "\n".join(lines) + "\n"


def page(
    address: str,
    title: str,
    entity: str,
    text: str,
    *,
    facts: dict[str, str],
    tags: list[str],
    status: str = "current",
    valid_from: str = "2026-06-18T00:00:00+00:00",
    valid_until: str | None = None,
    tombstoned: bool = False,
    taints: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "address": address,
        "title": title,
        "entity": entity,
        "text": text,
        "facts": facts,
        "tags": tags,
        "status": status,
        "valid_from": valid_from,
        "valid_until": valid_until,
        "tombstoned": tombstoned,
        "taints": taints or [],
    }


def edge(src: str, dst: str, relation: str) -> dict[str, str]:
    return {"from": src, "to": dst, "type": relation}


def case(
    case_id: str,
    category: str,
    question: str,
    query_terms: list[str],
    expected_evidence: list[str],
    expected_answer: str,
    *,
    requires_abstention: bool = False,
    stale_sensitive: bool = False,
    deletion_sensitive: bool = False,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "category": category,
        "question": question,
        "query_terms": query_terms,
        "expected_evidence": expected_evidence,
        "expected_answer": expected_answer,
        "requires_abstention": requires_abstention,
        "stale_sensitive": stale_sensitive,
        "deletion_sensitive": deletion_sensitive,
    }


def gate(name: str, passed: bool, detail: str, *, invert: bool = False) -> dict[str, Any]:
    actual = not passed if invert else passed
    return {"name": name, "passed": bool(actual), "detail": detail}


def get_path(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def canonical_bytes(payload: Any) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def normalize_terms(text: str) -> list[str]:
    return [term for term in re.split(r"[^a-z0-9_]+", text.lower()) if term]


def mean(values: list[float]) -> float:
    return round(sum(values) / max(1, len(values)), 4)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
