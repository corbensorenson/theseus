"""Same-seed evidence and control gate for STS-conditioned decoding.

The evidence side reads candidate manifests from the latest private closure and
checks whether STS-conditioned candidates changed the learned-token inventory
in a measurable way. The control side materializes metadata-only decoder
pressure for the next private closure. It never reads benchmark solutions or
runs public tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--private-candidates",
        default="reports/code_lm_private_candidates_private_pressure_private_recovery_train_once_fanout_v1.jsonl",
    )
    parser.add_argument(
        "--public-candidates",
        default="reports/student_code_candidates_private_pressure_private_recovery_train_once_fanout_v1.jsonl",
    )
    parser.add_argument("--decoder-gate", default="reports/decoder_v2_private_ablation_gate.json")
    parser.add_argument("--out", default="reports/sts_causal_decoder_ablation.json")
    parser.add_argument("--control-out", default="reports/sts_decoder_control_contract.json")
    parser.add_argument("--control-jsonl-out", default="reports/sts_decoder_control_rows.jsonl")
    parser.add_argument(
        "--control-binary-out",
        default="reports/sts_decoder_control_rows.f32bin",
        help="Compact binary vector sidecar for GPU STS retrieval/conditioning.",
    )
    args = parser.parse_args()

    private_rows = read_jsonl(ROOT / args.private_candidates)
    public_rows = read_jsonl(ROOT / args.public_candidates)
    decoder_gate = read_json(ROOT / args.decoder_gate)
    private_metrics = manifest_metrics(private_rows)
    public_metrics = manifest_metrics(public_rows)
    public_non_sts = public_metrics["groups"]["non_sts_learned_token"]
    public_sts = public_metrics["groups"]["sts_conditioned"]
    comparator_present = public_non_sts["row_count"] > 0
    summary = {
        "private": private_metrics,
        "public": public_metrics,
        "decoder_gate_ready": bool(
            nested(decoder_gate, "summary", "ready_for_public_calibration")
            or decoder_gate.get("ready_for_public_calibration")
        ),
        "decoder_public_no_admissible_task_rate": safe_float(
            nested(decoder_gate, "summary", "public_no_admissible_task_rate")
        ),
        "same_seed_non_sts_comparator_present": comparator_present,
        "sts_public_eligible_coverage_delta": (
            round(
                public_sts["eligible_task_coverage"] - public_non_sts["eligible_task_coverage"],
                6,
            )
            if comparator_present
            else None
        ),
        "sts_public_pass_rate_delta": (
            round(public_sts["verifier_pass_rate"] - public_non_sts["verifier_pass_rate"], 6)
            if comparator_present
            else None
        ),
        "sts_contract_public_task_coverage": public_metrics["groups"][
            "sts_contract_guided_token"
        ]["task_coverage"],
    }
    if comparator_present:
        summary["sts_candidate_distribution_delta"] = round(
            abs(summary["sts_public_eligible_coverage_delta"] or 0.0)
            + abs(summary["sts_public_pass_rate_delta"] or 0.0),
            6,
        )
        eligible_delta = summary["sts_public_eligible_coverage_delta"] or 0.0
        pass_delta = summary["sts_public_pass_rate_delta"] or 0.0
        summary["sts_positive_same_seed_lift"] = bool(
            eligible_delta >= 0.01 or pass_delta >= 0.01
        )
        summary["sts_coverage_non_regressive"] = bool(
            eligible_delta >= -0.02 and pass_delta >= -0.02
        )
        summary["sts_conditioning_regressed_candidate_coverage"] = bool(eligible_delta < -0.02)
    else:
        summary["sts_candidate_distribution_delta"] = None
        summary["sts_positive_same_seed_lift"] = False
        summary["sts_coverage_non_regressive"] = False
        summary["sts_conditioning_regressed_candidate_coverage"] = False
    control_contract, control_rows = materialize_control_contract(
        summary=summary,
        private_metrics=private_metrics,
        public_metrics=public_metrics,
        decoder_gate=decoder_gate,
        control_out=ROOT / args.control_out,
        control_jsonl_out=ROOT / args.control_jsonl_out,
    )
    write_json(ROOT / args.control_out, control_contract)
    write_jsonl(ROOT / args.control_jsonl_out, control_rows)
    binary_sidecar = write_control_binary_sidecar(ROOT / args.control_binary_out, control_rows)
    summary["sts_control_contract_path"] = rel(ROOT / args.control_out)
    summary["sts_control_rows_path"] = rel(ROOT / args.control_jsonl_out)
    summary["sts_control_binary_path"] = rel(binary_sidecar)
    summary["sts_control_rows_written"] = len(control_rows)
    summary["sts_control_consumer_count"] = len(control_contract["consumer_contract"]["consumers"])
    summary["sts_control_materialized"] = True
    summary["sts_control_contract_ready"] = bool(
        control_contract["consumer_contract"]["consumers"] and control_rows
    )
    gates = [
        gate("private_manifest_present", bool(private_rows), str(ROOT / args.private_candidates)),
        gate("public_manifest_present", bool(public_rows), str(ROOT / args.public_candidates)),
        gate(
            "sts_conditioned_candidates_observed",
            public_metrics["groups"]["sts_conditioned"]["row_count"] >= 16,
            public_metrics["groups"]["sts_conditioned"],
        ),
        gate(
            "sts_contract_guided_token_observed",
            public_metrics["groups"]["sts_contract_guided_token"]["row_count"] >= 8,
            public_metrics["groups"]["sts_contract_guided_token"],
        ),
        gate(
            "same_seed_non_sts_comparator_present",
            comparator_present,
            public_non_sts,
        ),
        gate(
            "sts_changes_candidate_distribution",
            comparator_present
            and (
                abs(summary["sts_public_eligible_coverage_delta"] or 0.0) >= 0.01
                or abs(summary["sts_public_pass_rate_delta"] or 0.0) >= 0.01
            ),
            {
                "eligible_coverage_delta": summary["sts_public_eligible_coverage_delta"],
                "pass_rate_delta": summary["sts_public_pass_rate_delta"],
            },
        ),
        gate(
            "sts_not_worse_than_non_sts_by_more_than_two_points",
            comparator_present and (summary["sts_public_pass_rate_delta"] or 0.0) >= -0.02,
            summary["sts_public_pass_rate_delta"],
        ),
        gate(
            "sts_eligible_coverage_non_regressive",
            bool(summary["sts_coverage_non_regressive"]),
            {
                "eligible_coverage_delta": summary["sts_public_eligible_coverage_delta"],
                "pass_rate_delta": summary["sts_public_pass_rate_delta"],
                "minimum_allowed_eligible_delta": -0.02,
                "rule": "STS may be a repair/control signal when worse, but not a GREEN causal improvement.",
            },
        ),
        gate(
            "sts_same_seed_positive_lift",
            bool(summary["sts_positive_same_seed_lift"]),
            {
                "eligible_coverage_delta": summary["sts_public_eligible_coverage_delta"],
                "pass_rate_delta": summary["sts_public_pass_rate_delta"],
                "minimum_positive_delta": 0.01,
                "rule": "Architecture promotion requires STS-on to beat same-seed STS-off on coverage or pass rate.",
            },
        ),
        gate(
            "public_no_admissible_below_unlock_threshold",
            (
                summary["decoder_public_no_admissible_task_rate"] is not None
                and summary["decoder_public_no_admissible_task_rate"] <= 0.25
            ),
            summary["decoder_public_no_admissible_task_rate"],
        ),
        gate(
            "sts_control_contract_written",
            bool(control_rows) and (ROOT / args.control_out).exists() and (ROOT / args.control_jsonl_out).exists(),
            {
                "control_contract_path": rel(ROOT / args.control_out),
                "control_rows_path": rel(ROOT / args.control_jsonl_out),
                "control_binary_path": rel(binary_sidecar),
                "control_rows_written": len(control_rows),
            },
        ),
        gate(
            "sts_control_binary_sidecar_written",
            binary_sidecar.exists() and binary_sidecar.stat().st_size > 24,
            {
                "control_binary_path": rel(binary_sidecar),
                "format": "THSTSVEC1 row_count/u32 dim/u32 then row_id_len/u16 row_id bytes and dim f32 values",
            },
        ),
        gate(
            "sts_control_has_named_consumers",
            bool(control_contract["consumer_contract"]["consumers"])
            and bool(control_contract["consumer_contract"]["effects"]),
            control_contract["consumer_contract"],
        ),
        gate(
            "sts_control_requests_same_seed_comparator_when_missing",
            comparator_present
            or bool(control_contract["decoder_hints"].get("force_same_seed_non_sts_comparator")),
            control_contract["decoder_hints"],
        ),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    report = {
        "policy": "project_theseus_sts_causal_decoder_ablation_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "ready_for_architecture_promotion": trigger_state == "GREEN"
        and bool(summary["decoder_gate_ready"]),
        "summary": summary,
        "gates": gates,
        "rules": {
            "no_public_solution_use": True,
            "same_seed_scope": "candidate manifests from the same completed closure are grouped by mode and scored without executing public tests",
            "promotion_rule": "STS must be observed, change candidate distribution, avoid regression, and pass the decoder/private transfer gates before public calibration unlocks",
            "control_rule": "Control rows are metadata-only decoder pressure and never contain public prompts, tests, answers, or template bodies.",
        },
        "control_contract": control_contract,
        "next_actions": next_actions(trigger_state, summary),
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def manifest_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_rows[task_id(row)].append(row)
    task_count = len(task_rows)
    groups = {
        "learned_token": lambda row: is_learned_token(row),
        "non_sts_learned_token": lambda row: is_learned_token(row) and not is_sts(row),
        "sts_conditioned": lambda row: is_learned_token(row) and is_sts(row),
        "contract_guided_token": lambda row: is_learned_token(row) and is_contract_guided(row),
        "sts_contract_guided_token": lambda row: is_learned_token(row)
        and is_sts(row)
        and is_contract_guided(row),
        "ast_or_interface_repair": lambda row: is_learned_token(row)
        and ("parser_ast_completion" in mode(row) or "interface_role_repair" in mode(row)),
    }
    group_metrics = {}
    for name, predicate in groups.items():
        selected = [row for row in rows if predicate(row)]
        task_ids = {task_id(row) for row in selected}
        eligible_task_ids = {task_id(row) for row in selected if verifier_passed(row)}
        group_metrics[name] = {
            "row_count": len(selected),
            "task_count": len(task_ids),
            "task_coverage": ratio(len(task_ids), task_count),
            "eligible_task_count": len(eligible_task_ids),
            "eligible_task_coverage": ratio(len(eligible_task_ids), task_count),
            "verifier_pass_rate": ratio(sum(verifier_passed(row) for row in selected), len(selected)),
            "mode_counts": Counter(mode(row) for row in selected).most_common(12),
        }
    no_admissible_tasks = set()
    for task, task_specific_rows in task_rows.items():
        has_learned_token = any(is_learned_token(row) for row in task_specific_rows)
        has_no_admissible_marker = any("no_admissible" in mode(row) for row in task_specific_rows)
        if has_no_admissible_marker and not has_learned_token:
            no_admissible_tasks.add(task)
    rejection_reasons = Counter()
    for row in rows:
        if "no_admissible" in mode(row):
            for reason, count in (row.get("candidate_rejection_counts") or {}).items():
                rejection_reasons[str(reason)] += int(count)
        for reason in row.get("decoder_contract_verifier_v1_reasons") or []:
            rejection_reasons[str(reason)] += 1
    return {
        "row_count": len(rows),
        "task_count": task_count,
        "no_admissible_task_count": len(no_admissible_tasks),
        "no_admissible_task_rate": ratio(len(no_admissible_tasks), task_count),
        "groups": group_metrics,
        "top_rejection_reasons": rejection_reasons.most_common(20),
    }


def mode(row: dict[str, Any]) -> str:
    return str(row.get("candidate_generation_mode") or row.get("mode") or "")


def task_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("source_task_id") or row.get("task_key") or "unknown")


def is_sts(row: dict[str, Any]) -> bool:
    return "sts_conditioned" in mode(row)


def is_contract_guided(row: dict[str, Any]) -> bool:
    row_mode = mode(row)
    return (
        "contract_guided_token_decoder" in row_mode
        or "contract_transduced_token_decoder" in row_mode
    )


def is_learned_token(row: dict[str, Any]) -> bool:
    row_mode = mode(row).lower()
    if any(
        token in row_mode
        for token in [
            "no_admissible",
            "prompt_program_decoder",
            "same_seed_non_sts_comparator",
            "skeleton",
            "prototype",
            "ngram",
            "semantic_plan",
            "native_sts_stream_expression",
        ]
    ):
        return False
    if bool(row.get("grammar_masked_learned_token_candidate")):
        return True
    loop = row.get("program_synthesis_loop_v1") if isinstance(row.get("program_synthesis_loop_v1"), dict) else {}
    decode = loop.get("decode_control") if isinstance(loop.get("decode_control"), dict) else {}
    return bool(
        row.get("full_body_token_candidate") is True
        and row.get("compositional_token_candidate") is True
        and decode.get("constrained_token_decode") is True
        and decode.get("parser_contract_mask") is True
        and not decode.get("template_or_memory_fallback")
        and ("token_decoder" in row_mode or "contract_transduced_token_decoder" in row_mode)
    )


def verifier_passed(row: dict[str, Any]) -> bool:
    return bool(row.get("decoder_contract_verifier_v1_passed"))


def next_actions(trigger_state: str, summary: dict[str, Any]) -> list[str]:
    if trigger_state == "GREEN" and summary.get("decoder_gate_ready"):
        return [
            "Run private_public_transfer_proof.py and keep public calibration locked until it also reports ready.",
            "If transfer proof is GREEN, allow exactly one bounded public 4-card calibration.",
        ]
    return [
        "Run a fresh train-once/fanout refresh that consumes the STS decoder-control rows and emits a same-seed non-STS comparator.",
        "Re-run decoder_v2_private_ablation_gate.py and this same-seed STS ablation before any public calibration.",
        "If STS remains weak, route the exact no-admissible/interface residual cluster to teacher-as-architect for one private experiment spec.",
    ]


def materialize_control_contract(
    *,
    summary: dict[str, Any],
    private_metrics: dict[str, Any],
    public_metrics: dict[str, Any],
    decoder_gate: dict[str, Any],
    control_out: Path,
    control_jsonl_out: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    public_reasons = top_reason_map(public_metrics)
    private_reasons = top_reason_map(private_metrics)
    targeted_families = targeted_capability_families(public_reasons, private_reasons, decoder_gate)
    comparator_missing = not bool(summary.get("same_seed_non_sts_comparator_present"))
    no_admissible_rate = safe_float(summary.get("decoder_public_no_admissible_task_rate"))
    sts_positive_lift = bool(summary.get("sts_positive_same_seed_lift"))
    sts_coverage_regressed = bool(summary.get("sts_conditioning_regressed_candidate_coverage"))
    control_objectives = [
        "emit_same_seed_non_sts_learned_token_comparator",
        "raise_sts_conditioned_candidate_task_coverage",
        "lower_no_admissible_candidate_rate",
        "prefer_exact_interface_and_return_shape_when_verifier_passes",
        "record_sts_vs_non_sts_rank_delta",
    ]
    if sts_coverage_regressed:
        control_objectives.insert(1, "repair_sts_candidate_coverage_before_promotion")
    rows: list[dict[str, Any]] = []
    for objective in control_objectives:
        rows.append(
            sts_control_row(
                objective=objective,
                targeted_families=targeted_families,
                public_reasons=public_reasons,
                comparator_missing=comparator_missing,
                no_admissible_rate=no_admissible_rate,
                sts_positive_lift=sts_positive_lift,
                sts_coverage_non_regressive=bool(summary.get("sts_coverage_non_regressive")),
                sts_coverage_regressed=sts_coverage_regressed,
            )
        )
    contract = {
        "policy": "project_theseus_sts_decoder_control_contract_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if rows else "YELLOW",
        "control_contract_path": rel(control_out),
        "control_rows_path": rel(control_jsonl_out),
        "control_rows_written": len(rows),
        "consumer_contract": {
            "consumers": [
                "code_lm_closure.run_sts_conditioning",
                "symliquid_state_engine.code_lm_closure_args",
                "decoder_v2_private_ablation_gate",
                "hive_work_board_executor",
                "agent_lane_transfer_gate",
            ],
            "effects": [
                "force same-seed non-STS learned-token comparator when absent",
                "bias STS conditioning rows toward no-admissible and interface residual families",
                "boost exact signature, argument-use, return-shape, branch-loop-local-state obligations",
                "demote STS preference when same-seed coverage regresses",
                "keep public calibration locked until decoder gate and private-public transfer proof are both GREEN",
            ],
            "promotion_evidence_required": "same_seed_sts_vs_non_sts_delta_after_completed_private_closure",
        },
        "decoder_hints": {
            "force_same_seed_non_sts_comparator": comparator_missing,
            "min_non_sts_comparator_task_coverage": 0.60,
            "min_sts_conditioned_task_coverage": 0.60,
            "prefer_sts_when_verifier_passes": sts_positive_lift and not sts_coverage_regressed,
            "rank_delta_ablation_required": True,
            "no_admissible_rate_floor": 0.25,
            "current_public_no_admissible_task_rate": no_admissible_rate,
            "sts_positive_same_seed_lift": sts_positive_lift,
            "sts_coverage_non_regressive": bool(summary.get("sts_coverage_non_regressive")),
            "sts_conditioning_regressed_candidate_coverage": sts_coverage_regressed,
            "min_positive_sts_delta_for_promotion": 0.01,
            "public_calibration_locked": True,
        },
        "targeted_capability_families": targeted_families,
        "top_public_rejection_reasons": public_metrics.get("top_rejection_reasons") or [],
        "top_private_rejection_reasons": private_metrics.get("top_rejection_reasons") or [],
        "source_metrics": {
            "same_seed_non_sts_comparator_present": summary.get("same_seed_non_sts_comparator_present"),
            "sts_public_eligible_coverage_delta": summary.get("sts_public_eligible_coverage_delta"),
            "sts_public_pass_rate_delta": summary.get("sts_public_pass_rate_delta"),
            "sts_candidate_distribution_delta": summary.get("sts_candidate_distribution_delta"),
            "decoder_gate_ready": summary.get("decoder_gate_ready"),
        },
        "safety": {
            "public_benchmark_solutions_included": False,
            "public_tests_included": False,
            "canonical_solution_exported": False,
            "raw_public_prompt_or_tests_copied": False,
            "template_body_training": False,
            "training_use_state": "decoder_control_policy_only_not_code_answer_training",
        },
    }
    return contract, rows


def sts_control_row(
    *,
    objective: str,
    targeted_families: list[str],
    public_reasons: dict[str, int],
    comparator_missing: bool,
    no_admissible_rate: float | None,
    sts_positive_lift: bool,
    sts_coverage_non_regressive: bool,
    sts_coverage_regressed: bool,
) -> dict[str, Any]:
    prefer_sts = sts_positive_lift and not sts_coverage_regressed
    digest = hashlib.sha256(
        json.dumps(
            {
                "objective": objective,
                "families": targeted_families,
                "reasons": public_reasons,
                "comparator_missing": comparator_missing,
                "no_admissible_rate": no_admissible_rate,
                "sts_positive_lift": sts_positive_lift,
                "sts_coverage_non_regressive": sts_coverage_non_regressive,
                "sts_coverage_regressed": sts_coverage_regressed,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    family_text = ", ".join(targeted_families[:8]) or "interface_and_return_shape"
    reason_text = ", ".join(f"{name}:{count}" for name, count in list(public_reasons.items())[:8])
    return {
        "row_id": f"sts_decoder_control_{digest}",
        "policy": "project_theseus_sts_decoder_control_row_v1",
        "dataset_id": "dataset.sts_decoder_control.v1",
        "source_type": "sts_decoder_control_contract",
        "split": "control",
        "benchmark_evidence_level": "decoder_control_metadata_only_not_code_answer_training",
        "training_use_state": "decoder_control_policy_only_not_code_answer_training",
        "objective": objective,
        "missing_capability_family": targeted_families[0] if targeted_families else "interface_and_return_shape",
        "targeted_capability_families": targeted_families,
        "candidate_rejection_reason_counts": public_reasons,
        "force_same_seed_non_sts_comparator": comparator_missing,
        "sts_positive_same_seed_lift": sts_positive_lift,
        "sts_coverage_non_regressive": sts_coverage_non_regressive,
        "sts_conditioning_regressed_candidate_coverage": sts_coverage_regressed,
        "prefer_sts_when_verifier_passes": prefer_sts,
        "decoder_control_effect": (
            "allow_sts_preference_after_positive_same_seed_lift"
            if prefer_sts
            else "demote_sts_preference_until_positive_same_seed_lift"
        ),
        "public_no_admissible_task_rate": no_admissible_rate,
        "answer": (
            f"decoder control objective={objective}; target_families={family_text}; "
            f"rejection_reasons={reason_text or 'none'}; "
            f"prefer_sts_when_verifier_passes={str(prefer_sts).lower()}; "
            f"sts_positive_same_seed_lift={str(sts_positive_lift).lower()}; "
            f"sts_coverage_non_regressive={str(sts_coverage_non_regressive).lower()}; "
            f"sts_conditioning_regressed_candidate_coverage={str(sts_coverage_regressed).lower()}; "
            "emit learned-token candidates and comparator metadata only"
        ),
        "visible_task_only": True,
        "public_benchmark_solutions_included": False,
        "public_tests_included": False,
        "canonical_solution_exported": False,
        "raw_public_prompt_or_tests_copied": False,
        "public_benchmark_training_data_used": False,
    }


def top_reason_map(*metrics: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for metric in metrics:
        for item in metric.get("top_rejection_reasons") or []:
            if not isinstance(item, (list, tuple)) or not item:
                continue
            try:
                counts[str(item[0])] += int(item[1])
            except (TypeError, ValueError, IndexError):
                counts[str(item[0])] += 1
    return dict(counts.most_common(16))


def targeted_capability_families(
    public_reasons: dict[str, int],
    private_reasons: dict[str, int],
    decoder_gate: dict[str, Any],
) -> list[str]:
    text = " ".join(list(public_reasons) + list(private_reasons)).lower()
    public_missing = nested(decoder_gate, "summary", "public_no_admissible_top_missing_capability_families")
    families: list[str] = []
    if isinstance(public_missing, dict):
        families.extend(str(name) for name, _ in Counter(public_missing).most_common(8))
    elif isinstance(public_missing, list):
        families.extend(str(row[0] if isinstance(row, (list, tuple)) and row else row) for row in public_missing[:8])
    keyword_families = [
        ("interface", "interface_fidelity"),
        ("signature", "interface_fidelity"),
        ("argument", "argument_role_mapping"),
        ("return", "return_shape_contract"),
        ("branch", "branch_skeleton"),
        ("loop", "loop_skeleton"),
        ("local", "local_state_tracking"),
        ("recursive", "recursive_or_nested_structure"),
        ("nested", "recursive_or_nested_structure"),
        ("list", "list_transform_body"),
        ("string", "string_transform_body"),
        ("json", "json_structured_io"),
        ("csv", "csv_structured_io"),
        ("archive", "archive_path_file_transform"),
        ("path", "archive_path_file_transform"),
        ("url", "url_encoding_transform"),
        ("system", "system_info_dict"),
        ("syntax", "parser_ast_completion"),
        ("parse", "parser_ast_completion"),
        ("no_admissible", "candidate_coverage_recovery"),
    ]
    for keyword, family in keyword_families:
        if keyword in text and family not in families:
            families.append(family)
    for fallback in [
        "candidate_coverage_recovery",
        "interface_fidelity",
        "return_shape_contract",
        "parser_ast_completion",
        "branch_loop_local_skeleton",
    ]:
        if fallback not in families:
            families.append(fallback)
    return families[:12]


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def ratio(num: int | float, denom: int | float) -> float:
    return round(float(num) / float(denom), 6) if denom else 0.0


def nested(payload: Any, *keys: str) -> Any:
    cur = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_control_binary_sidecar(path: Path, rows: list[dict[str, Any]], dim: int = 64) -> Path:
    """Write a compact deterministic STS control matrix for GPU retrieval.

    The JSONL stays as the auditable source of truth; this sidecar is the
    training-loop format so a CUDA worker can load control vectors without
    reparsing JSON or string-heavy metadata.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    dim = max(8, int(dim))
    with path.open("wb") as handle:
        handle.write(b"THSTSVEC1")
        handle.write(struct.pack("<II", len(rows), dim))
        for row in rows:
            row_id = str(row.get("row_id") or stable_row_id(row)).encode("utf-8")[:65535]
            vector = sts_control_vector(row, dim)
            handle.write(struct.pack("<H", len(row_id)))
            handle.write(row_id)
            handle.write(struct.pack(f"<{dim}f", *vector))
    return path


def sts_control_vector(row: dict[str, Any], dim: int) -> list[float]:
    vector = [0.0] * dim
    weighted_terms: list[tuple[str, float]] = [
        (str(row.get("objective") or ""), 2.0),
        (str(row.get("missing_capability_family") or ""), 1.6),
        ("force_same_seed_non_sts_comparator", 1.1 if row.get("force_same_seed_non_sts_comparator") else 0.2),
    ]
    for family in row.get("targeted_capability_families") or []:
        weighted_terms.append((f"family:{family}", 1.25))
    for reason, count in (row.get("candidate_rejection_reason_counts") or {}).items():
        try:
            weight = min(4.0, 0.25 + float(count) ** 0.5)
        except (TypeError, ValueError):
            weight = 0.5
        weighted_terms.append((f"reason:{reason}", weight))
    rate = row.get("public_no_admissible_task_rate")
    if isinstance(rate, (int, float)):
        weighted_terms.append(("public_no_admissible_task_rate", float(rate) * 2.0))
    for term, weight in weighted_terms:
        add_hashed_feature(vector, term, weight)
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


def add_hashed_feature(vector: list[float], term: str, weight: float) -> None:
    if not term or weight == 0.0:
        return
    digest = hashlib.sha256(term.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "little") % len(vector)
    sign = 1.0 if digest[4] & 1 == 0 else -1.0
    vector[idx] += float(weight) * sign


def stable_row_id(row: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"sts_decoder_control_{digest}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
