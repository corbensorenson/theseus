#!/usr/bin/env python3
"""Freeze and evaluate prompt-only KERC K4 interaction interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "kerc_k4_interaction_qualification_v2.json"
POLICY = "project_theseus_kerc_k4_interaction_qualification_v2"
PACKET_POLICY = "project_theseus_kerc_k4_prompt_packet_v2"


def resolve(value: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_id(*values: Any) -> str:
    body = json.dumps(values, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def case_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id") or ""): row for row in config.get("cases") or []}


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = config.get("kerc_k4_qualification")
    if not isinstance(contract, dict) or contract.get("policy") != POLICY:
        raise ValueError("K4 qualification contract missing or invalid")
    if contract.get("state") != "FROZEN_BEFORE_CANDIDATE_OUTPUTS":
        raise ValueError("K4 qualification must freeze before candidate outputs")
    cases = case_map(config)
    selected = [str(value) for value in contract.get("selected_case_ids") or []]
    if len(selected) != len(set(selected)) or any(value not in cases for value in selected):
        raise ValueError("K4 selected cases are missing or duplicated")
    if len(selected) < int((contract.get("acceptance") or {}).get("minimum_case_count") or 0):
        raise ValueError("K4 selected case count is below its frozen minimum")
    seeds = contract.get("model_seeds") or []
    if len(seeds) != len(set(seeds)) or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds):
        raise ValueError("K4 model seeds are invalid")
    required = set(contract.get("required_interventions") or [])
    if required != {
        "context_present",
        "context_withheld",
        "context_shuffled",
        "wrong_user",
        "stale_state",
        "expansion_replay",
    }:
        raise ValueError("K4 intervention set drifted")
    donors = contract.get("wrong_user_donor_by_case") or {}
    forbidden = contract.get("wrong_user_forbidden_terms_by_case") or {}
    if set(donors) != set(selected) or set(forbidden) != set(selected):
        raise ValueError("K4 wrong-user controls are incomplete")
    if any(donors.get(case_id) not in cases or donors.get(case_id) == case_id for case_id in selected):
        raise ValueError("K4 wrong-user donors are invalid")
    stale = set(contract.get("stale_state_case_ids") or [])
    if stale != set(selected):
        raise ValueError("K4 stale-state cases are invalid")
    if any(len(cases[case_id].get("turns") or []) < 3 for case_id in selected):
        raise ValueError("K4 shuffled/stale controls require at least three turns per case")
    if contract.get("prompt_packet_policy") != PACKET_POLICY:
        raise ValueError("K4 prompt packet policy drifted")
    if set(contract.get("candidate_visible_fields") or []) != {"prompt"}:
        raise ValueError("K4 model-visible fields must contain only the prompt")
    if set(contract.get("packet_row_fields") or []) != {"opaque_id", "prompt"}:
        raise ValueError("K4 packet row fields drifted")
    if set(contract.get("controller_visible_fields") or []) != {
        "checkpoint_seed",
        "opaque_id",
    }:
        raise ValueError("K4 controller-visible fields drifted")
    source = contract.get("source_provenance") or {}
    if (
        not str(source.get("source_family_id") or "")
        or source.get("admitted_to_training") is not False
        or source.get("training_use_forbidden") is not True
    ):
        raise ValueError("K4 source-family provenance is incomplete")
    if not str(contract.get("surface_id") or ""):
        raise ValueError("K4 surface identity is missing")
    if contract.get("surface_role") != "fresh_private_qualification":
        raise ValueError("K4 surface role must be fresh private qualification")
    if contract.get("consumption_registry") != (
        "reports/private_functional_consumption_registry.jsonl"
    ):
        raise ValueError("K4 exact-once consumption registry drifted")
    return contract


def history_and_query(case: dict[str, Any]) -> tuple[list[str], str]:
    turns = case.get("turns") or []
    if len(turns) < 2 or any(not isinstance(row, dict) or not str(row.get("user") or "") for row in turns):
        raise ValueError(f"K4 case lacks two complete turns: {case.get('id')}")
    return [str(row["user"]) for row in turns[:-1]], str(turns[-1]["user"])


def render_history(history: list[str], query: str) -> str:
    lines = ["Conversation history:"]
    lines.extend(f"User turn {index + 1}: {value}" for index, value in enumerate(history))
    lines.extend(["Current user:", query])
    return "\n".join(lines)


def intervention_prompts(
    case_id: str,
    *,
    cases: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, str]:
    history, query = history_and_query(cases[case_id])
    donor_id = str((contract.get("wrong_user_donor_by_case") or {})[case_id])
    donor_history, _donor_query = history_and_query(cases[donor_id])
    prompts = {
        "context_present": render_history(history, query),
        "context_withheld": "Current user:\n" + query,
        "context_shuffled": render_history(list(reversed(history)), query),
        "wrong_user": render_history(donor_history, query),
        "expansion_replay": "Expanded interaction state:\n"
        + json.dumps(
            {"ordered_user_history": history, "current_user": query},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }
    if case_id in set(contract.get("stale_state_case_ids") or []):
        stale_history = history[:-1]
        if not stale_history:
            raise ValueError(f"K4 stale-state case has no superseded history: {case_id}")
        prompts["stale_state"] = render_history(stale_history, query)
    return prompts


def training_overlap_audit(
    config: dict[str, Any], *, root: Path
) -> dict[str, Any]:
    contract = validate_contract(config)
    cases = case_map(config)
    source_family_id = str(
        (contract.get("source_provenance") or {}).get("source_family_id") or ""
    )
    needles = {
        case_id: [
            normalize_overlap_text(str(turn["user"]))
            for turn in cases[case_id].get("turns") or []
        ]
        for case_id in contract.get("selected_case_ids") or []
    }
    searched = []
    overlaps = []
    source_family_mentions = []
    for directory_name in ("data/training_data", "data/training_sources"):
        directory = root / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".txt", ".md"}:
                continue
            searched.append(path.relative_to(root).as_posix())
            relative_path = path.relative_to(root).as_posix()
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line_number, line in enumerate(handle, 1):
                    normalized = normalize_overlap_text(line.replace("\\n", " "))
                    if source_family_id.casefold() in line.casefold():
                        source_family_mentions.append(
                            {"path": relative_path, "line_number": line_number}
                        )
                    for case_id, values in needles.items():
                        for turn_index, needle in enumerate(values):
                            if needle and needle in normalized:
                                overlaps.append(
                                    {
                                        "case_id": case_id,
                                        "turn_index": turn_index,
                                        "path": relative_path,
                                        "line_number": line_number,
                                    }
                                )
    provenance = contract.get("source_provenance") or {}
    provenance_disjoint = (
        provenance.get("admitted_to_training") is False
        and provenance.get("training_use_forbidden") is True
        and not source_family_mentions
    )
    return {
        "searched_file_count": len(searched),
        "normalized_full_turn_overlap_count": len(overlaps),
        "overlaps": overlaps,
        "source_family_id": source_family_id,
        "source_family_training_root_mention_count": len(source_family_mentions),
        "source_family_mentions": source_family_mentions,
        "declared_provenance_disjoint": provenance_disjoint,
        "source_family_disjoint": provenance_disjoint and not overlaps,
        "scope": (
            "streamed normalized full-turn scan across registered local training-data "
            "roots plus explicit evaluation-family provenance separation"
        ),
    }


def normalize_overlap_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def build_prompt_packet(
    config: dict[str, Any], *, config_path: Path = DEFAULT_CONFIG, root: Path = ROOT
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    contract = validate_contract(config)
    cases = case_map(config)
    config_digest = sha256(config_path)
    hidden_index: dict[str, dict[str, Any]] = {}
    checkpoint_groups = []
    for seed in contract.get("model_seeds") or []:
        rows = []
        for case_id in contract.get("selected_case_ids") or []:
            prompts = intervention_prompts(case_id, cases=cases, contract=contract)
            for intervention, prompt in sorted(prompts.items()):
                opaque_id = stable_id(POLICY, config_digest, seed, case_id, intervention)
                rows.append(
                    {"opaque_id": opaque_id, "prompt": prompt}
                )
                hidden_index[opaque_id] = {
                    "case_id": case_id,
                    "model_seed": int(seed),
                    "intervention": intervention,
                    "expected": cases[case_id]["turns"][-1],
                    "wrong_user_forbidden_terms": list(
                        (contract.get("wrong_user_forbidden_terms_by_case") or {}).get(case_id) or []
                    ),
                }
        checkpoint_groups.append({"checkpoint_seed": int(seed), "rows": rows})
    packet_fields = set(contract.get("packet_row_fields") or [])
    if any(
        set(row) != packet_fields
        for group in checkpoint_groups
        for row in group["rows"]
    ):
        raise ValueError("candidate packet contains undeclared fields")
    overlap = training_overlap_audit(config, root=root)
    packet = {
        "policy": PACKET_POLICY,
        "surface_id": contract["surface_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "path": config_path.relative_to(root).as_posix() if config_path.is_relative_to(root) else str(config_path),
            "sha256": config_digest,
        },
        "row_count": len(hidden_index),
        "checkpoint_groups": checkpoint_groups,
        "consumption_registry": contract["consumption_registry"],
        "information_flow": {
            "candidate_visible_fields": list(contract.get("candidate_visible_fields") or []),
            "packet_row_fields": sorted(packet_fields),
            "controller_visible_fields": list(
                contract.get("controller_visible_fields") or []
            ),
            "candidate_forbidden_fields": list(contract.get("candidate_forbidden_fields") or []),
            "answer_identifying_metadata_exposed": False,
            "hidden_index_serialized": False,
        },
        "training_overlap_audit": overlap,
        "boundaries": dict(contract.get("boundaries") or {}),
    }
    return packet, hidden_index


def score_output(output: str, expected: dict[str, Any]) -> float:
    lowered = output.casefold()
    if any(
        str(term).casefold() in lowered
        for term in expected.get("forbidden_terms") or []
    ):
        return 0.0
    checks = []
    for term in expected.get("expected_terms") or []:
        checks.append(str(term).casefold() in lowered)
    for group in expected.get("expected_any") or []:
        checks.append(any(str(term).casefold() in lowered for term in group))
    return sum(checks) / len(checks) if checks else 0.0


def load_outputs(path: Path) -> tuple[dict[str, str], list[str]]:
    outputs: dict[str, str] = {}
    faults = []
    if not path.is_file():
        return outputs, ["candidate_outputs_missing"]
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            faults.append(f"candidate_output_invalid_json:{line_number}")
            continue
        if not isinstance(row, dict) or set(row) != {"opaque_id", "output"}:
            faults.append(f"candidate_output_schema_invalid:{line_number}")
            continue
        opaque_id = str(row.get("opaque_id") or "")
        output = str(row.get("output") or "")
        if not opaque_id or opaque_id in outputs:
            faults.append(f"candidate_output_identity_invalid:{line_number}")
            continue
        outputs[opaque_id] = output
    return outputs, faults


def confidence_interval_lower(values: list[float]) -> float:
    if len(values) < 2:
        return float("-inf")
    return statistics.mean(values) - 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def audit_candidate_producer(
    contract: dict[str, Any],
    *,
    packet_path: Path,
    outputs_path: Path,
    expected_ids: set[str],
    root: Path,
) -> tuple[dict[str, Any], list[str]]:
    report_path = resolve(contract["candidate_producer_report"], root=root)
    guard_path = resolve(contract["candidate_producer_guard_receipt"], root=root)
    report = read_json(report_path) if report_path.is_file() else {}
    guard = read_json(guard_path) if guard_path.is_file() else {}
    faults: list[str] = []
    if report.get("policy") != "project_theseus_kerc_k4_candidate_producer_v2":
        faults.append("candidate_producer_report_missing_or_invalid")
    if report.get("trigger_state") != "GREEN":
        faults.append("candidate_producer_execution_not_green")
    if report.get("surface_id") != contract.get("surface_id"):
        faults.append("candidate_producer_surface_identity_mismatch")
    if (report.get("packet") or {}).get("sha256") != (
        sha256(packet_path) if packet_path.is_file() else ""
    ):
        faults.append("candidate_producer_packet_binding_mismatch")
    if (report.get("outputs") or {}).get("sha256") != (
        sha256(outputs_path) if outputs_path.is_file() else ""
    ):
        faults.append("candidate_producer_output_binding_mismatch")
    if int(report.get("row_count") or -1) != len(expected_ids):
        faults.append("candidate_producer_row_count_mismatch")
    if report.get("candidate_visible_fields") != ["prompt"]:
        faults.append("candidate_producer_model_view_mismatch")
    if report.get("answer_identifying_metadata_exposed") is not False:
        faults.append("candidate_producer_answer_metadata_exposed")
    for key in (
        "public_training_rows",
        "external_inference_calls",
        "fallback_template_router_tool_credit",
    ):
        if int(report.get(key) or 0) != 0:
            faults.append(f"candidate_producer_nonzero_boundary:{key}")
    observations = report.get("observations") or []
    if (
        len(observations) != len(expected_ids)
        or {str(row.get("opaque_id") or "") for row in observations} != expected_ids
    ):
        faults.append("candidate_producer_observation_identity_mismatch")
    else:
        loaded, output_faults = load_outputs(outputs_path)
        if output_faults:
            faults.append("candidate_producer_output_schema_fault")
        for row in observations:
            opaque_id = str(row["opaque_id"])
            expected_hash = hashlib.sha256(
                loaded.get(opaque_id, "").encode("utf-8")
            ).hexdigest()
            if row.get("output_sha256") != expected_hash:
                faults.append("candidate_producer_observation_hash_mismatch")
                break
    source_artifacts = report.get("source_artifacts") or {}
    required_sources = {
        "producer": ROOT / "scripts/kerc_k4_candidate_producer.py",
        "generator": ROOT / "scripts/moecot_language_arm_training.py",
    }
    for name, path in required_sources.items():
        row = source_artifacts.get(name) or {}
        if (
            row.get("path")
            != (
                path.relative_to(root).as_posix()
                if path.is_relative_to(root)
                else str(path)
            )
            or row.get("sha256") != sha256(path)
        ):
            faults.append(f"candidate_producer_source_binding_mismatch:{name}")
    if guard.get("passed") is not True or int(guard.get("returncode") or 0) != 0:
        faults.append("candidate_producer_guard_not_green")
    if float(guard.get("maximum_swapout_growth_mib") or 0.0) > float(
        (guard.get("limits") or {}).get("maximum_swapout_growth_mib") or 0.0
    ):
        faults.append("candidate_producer_guard_swap_limit_exceeded")
    return {
        "passed": not faults,
        "report": {
            "path": (
                report_path.relative_to(root).as_posix()
                if report_path.is_relative_to(root)
                else str(report_path)
            ),
            "sha256": sha256(report_path) if report_path.is_file() else "",
        },
        "guard": {
            "path": (
                guard_path.relative_to(root).as_posix()
                if guard_path.is_relative_to(root)
                else str(guard_path)
            ),
            "sha256": sha256(guard_path) if guard_path.is_file() else "",
        },
        "candidate_flags_recomputed": True,
        "output_identity_and_hashes_recomputed": True,
    }, faults


def evaluate(
    config: dict[str, Any],
    *,
    config_path: Path,
    packet_path: Path,
    outputs_path: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    contract = validate_contract(config)
    packet, hidden = build_prompt_packet(config, config_path=config_path, root=root)
    recorded_packet = read_json(packet_path) if packet_path.is_file() else {}
    outputs, faults = load_outputs(outputs_path)
    recorded_replay = {key: value for key, value in recorded_packet.items() if key != "created_utc"}
    expected_replay = {key: value for key, value in packet.items() if key != "created_utc"}
    if recorded_replay != expected_replay:
        faults.append("prompt_packet_missing_stale_or_tampered")
    expected_ids = set(hidden)
    if set(outputs) != expected_ids:
        faults.append("candidate_output_identity_set_mismatch")
    producer_audit, producer_faults = audit_candidate_producer(
        contract,
        packet_path=packet_path,
        outputs_path=outputs_path,
        expected_ids=expected_ids,
        root=root,
    )
    faults.extend(producer_faults)
    result_rows = []
    donor_leaks = 0
    wrong_user_count = 0
    forbidden_term_outputs = 0
    for opaque_id, metadata in hidden.items():
        output = outputs.get(opaque_id, "")
        score = score_output(output, metadata["expected"]) if output else 0.0
        forbidden_terms = [
            term
            for term in metadata["expected"].get("forbidden_terms") or []
            if str(term).casefold() in output.casefold()
        ]
        forbidden_term_outputs += bool(forbidden_terms)
        leaked = []
        if metadata["intervention"] == "wrong_user":
            wrong_user_count += 1
            leaked = [
                term
                for term in metadata["wrong_user_forbidden_terms"]
                if str(term).casefold() in output.casefold()
            ]
            donor_leaks += bool(leaked)
        result_rows.append(
            {
                **metadata,
                "opaque_id": opaque_id,
                "score": score,
                "donor_leak_terms": leaked,
                "forbidden_terms_present": forbidden_terms,
            }
        )
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in result_rows:
        by_seed.setdefault(int(row["model_seed"]), []).append(row)
    seed_effects = []
    for seed, rows in sorted(by_seed.items()):
        present = [row["score"] for row in rows if row["intervention"] == "context_present"]
        controls = [
            row["score"]
            for row in rows
            if row["intervention"] in {"context_withheld", "context_shuffled", "wrong_user", "stale_state"}
        ]
        seed_effects.append(statistics.mean(present) - statistics.mean(controls))
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    case_effects = []
    for case_id, rows in sorted(by_case.items()):
        present = [
            row["score"] for row in rows if row["intervention"] == "context_present"
        ]
        controls = [
            row["score"]
            for row in rows
            if row["intervention"]
            in {"context_withheld", "context_shuffled", "wrong_user", "stale_state"}
        ]
        case_effects.append(
            {
                "case_id": case_id,
                "effect": statistics.mean(present) - statistics.mean(controls),
            }
        )
    mean_by_intervention = {
        intervention: statistics.mean(
            row["score"] for row in result_rows if row["intervention"] == intervention
        )
        for intervention in contract.get("required_interventions") or []
        if any(row["intervention"] == intervention for row in result_rows)
    }
    acceptance = contract.get("acceptance") or {}
    seed_effect_lower = confidence_interval_lower(seed_effects)
    case_effect_values = [float(row["effect"]) for row in case_effects]
    case_effect_lower = confidence_interval_lower(case_effect_values)
    effect_lower = min(seed_effect_lower, case_effect_lower)
    leakage_rate = donor_leaks / wrong_user_count if wrong_user_count else 1.0
    forbidden_output_rate = (
        forbidden_term_outputs / len(result_rows) if result_rows else 1.0
    )
    missing_outputs = len(expected_ids - set(outputs))
    control_coverage = {
        intervention: intervention in mean_by_intervention
        for intervention in contract.get("required_interventions") or []
    }
    predicates = {
        "seed_count": len(by_seed) >= int(acceptance.get("minimum_seed_count") or 0),
        "case_count": len(set(row["case_id"] for row in result_rows)) >= int(acceptance.get("minimum_case_count") or 0),
        "all_controls_present": all(control_coverage.values()),
        "context_present": mean_by_intervention.get("context_present", 0.0) >= float(acceptance.get("minimum_context_present_score") or 0),
        "expansion_replay": mean_by_intervention.get("expansion_replay", 0.0) >= float(acceptance.get("minimum_expansion_replay_score") or 0),
        "causal_effect": effect_lower > float(acceptance.get("minimum_effect_confidence_interval_lower") or 0),
        "weak_case_effect": min(case_effect_values or [float("-inf")])
        >= float(acceptance.get("minimum_weak_case_effect") or 0),
        "weak_seed_effect": min(seed_effects or [float("-inf")])
        >= float(acceptance.get("minimum_weak_seed_effect") or 0),
        "wrong_user_leakage": leakage_rate <= float(acceptance.get("maximum_wrong_user_donor_leakage_rate") or 0),
        "forbidden_term_output_rate": forbidden_output_rate
        <= float(acceptance.get("maximum_forbidden_term_output_rate") or 0),
        "outputs_complete": missing_outputs <= int(acceptance.get("maximum_missing_output_count") or 0),
        "source_disjoint": bool((packet.get("training_overlap_audit") or {}).get("source_family_disjoint")),
    }
    if not all(predicates.values()):
        faults.append("acceptance_predicates_failed")
    source_paths = {
        "config": config_path,
        "implementation": Path(__file__),
        "prompt_packet": packet_path,
        "candidate_outputs": outputs_path,
    }
    source_artifacts = {
        name: {
            "path": path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),
            "sha256": sha256(path),
        }
        for name, path in source_paths.items()
        if path.is_file()
    }
    ready = not faults
    return {
        "policy": POLICY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_state": "GREEN" if ready else "RED",
        "source_artifacts": source_artifacts,
        "source_family_disjoint": predicates["source_disjoint"],
        "seed_count": len(by_seed),
        "case_count": len(set(row["case_id"] for row in result_rows)),
        "controls": control_coverage,
        "effect": {
            "seed_effects": seed_effects,
            "case_effects": case_effects,
            "mean": statistics.mean(seed_effects) if seed_effects else None,
            "seed_confidence_interval_lower": seed_effect_lower,
            "case_confidence_interval_lower": case_effect_lower,
            "confidence_interval_lower": effect_lower,
            "confidence_rule": "minimum_of_seed_cluster_and_case_cluster_95pct_normal_lower_bounds",
        },
        "mean_score_by_intervention": mean_by_intervention,
        "wrong_user_donor_leakage_rate": leakage_rate,
        "forbidden_term_output_rate": forbidden_output_rate,
        "missing_output_count": missing_outputs,
        "acceptance_predicates": predicates,
        "independent_audit": {
            "passed": ready,
            "producer_evaluator_separated": True,
            "candidate_flags_recomputed": True,
            "candidate_producer": producer_audit,
        },
        "anti_cheating": {"answer_identifying_metadata_exposed": False},
        "public_benchmark_prompts_used_for_training": 0,
        "runtime_external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
        "faults": faults,
        "non_claims": [
            "Prompt materialization and RED evaluation are not K4 behavioral evidence.",
            "Only complete direct candidate outputs from independently trained seeds can satisfy this gate.",
            "The existing deterministic conversation baseline receives no learned-generation credit.",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--outputs", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.materialize == args.evaluate:
        parser.error("choose exactly one of --materialize or --evaluate")
    config_path = resolve(args.config)
    config = read_json(config_path)
    contract = validate_contract(config)
    packet_path = resolve(str(contract["prompt_packet"]))
    if args.materialize:
        packet, _hidden = build_prompt_packet(config, config_path=config_path)
        output = resolve(args.out) if args.out else packet_path
        write_json(output, packet)
        print(json.dumps({"trigger_state": "GREEN", "row_count": packet["row_count"], "out": str(output)}, indent=2))
        return 0
    outputs_path = resolve(args.outputs or str(contract["candidate_outputs"]))
    report = evaluate(
        config,
        config_path=config_path,
        packet_path=packet_path,
        outputs_path=outputs_path,
    )
    output = resolve(args.out or str(contract["evaluation_report"]))
    write_json(output, report)
    print(json.dumps({"trigger_state": report["trigger_state"], "faults": report["faults"], "out": str(output)}, indent=2))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
