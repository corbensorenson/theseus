#!/usr/bin/env python3
"""Issue the terminal disposition for the frozen local-8B qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = (
    ROOT / "reports" / "core_evidence_local_8b_qualification_evaluation.json"
)
CANDIDATES = (
    ROOT / "reports" / "core_evidence_local_8b_qualification_candidates.json"
)
FREEZE = ROOT / "configs" / "core_evidence_local_8b_qualification_freeze.json"
EVALUATOR_MANIFEST = (
    ROOT / "configs" / "core_evidence_local_8b_qualification_evaluator.json"
)
OUT = ROOT / "reports" / "core_evidence_local_8b_qualification_disposition.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", default=str(EVALUATION))
    parser.add_argument("--candidates", default=str(CANDIDATES))
    parser.add_argument("--freeze", default=str(FREEZE))
    parser.add_argument("--evaluator-manifest", default=str(EVALUATOR_MANIFEST))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    report = build(
        Path(args.evaluation),
        Path(args.candidates),
        Path(args.freeze),
        Path(args.evaluator_manifest),
    )
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "disposition": report["disposition"],
        "qualification_passed": report["qualification_passed"],
        "attempted": report["observed"]["attempted"],
        "useful": report["observed"]["useful"],
        "unsafe": report["observed"]["unsafe"],
        "rollback_verified": report["observed"]["rollback_verified"],
    }, indent=2, sort_keys=True))
    return 0


def build(
    evaluation_path: Path,
    candidates_path: Path,
    freeze_path: Path,
    evaluator_manifest_path: Path,
) -> dict[str, Any]:
    evaluation = read_json(evaluation_path)
    candidates = read_json(candidates_path)
    freeze = read_json(freeze_path)
    evaluator = read_json(evaluator_manifest_path)
    validate_inputs(
        evaluation,
        candidates,
        freeze,
        evaluation_path,
        candidates_path,
        evaluator_manifest_path,
    )
    denominators = dict(evaluation["denominators"])
    attempted = int(denominators["attempted"])
    useful = int(denominators["useful"])
    family_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempted": 0, "useful": 0}
    )
    families = {
        task["opaque_task_id"]: task["family"] for task in evaluator["tasks"]
    }
    for row in evaluation["tasks"]:
        family = families[row["opaque_task_id"]]
        family_counts[family]["attempted"] += int(row["attempted"])
        family_counts[family]["useful"] += int(row["useful"])
    family_rates = {
        family: (
            values["useful"] / values["attempted"]
            if values["attempted"] else 0.0
        )
        for family, values in family_counts.items()
    }
    useful_rate = useful / attempted if attempted else 0.0
    weakest_rate = min(family_rates.values(), default=0.0)
    floor = freeze["competence_floor"]
    passed = bool(
        attempted >= int(floor["minimum_attempted_tasks"])
        and useful_rate >= float(floor["minimum_useful_rate"])
        and weakest_rate >= float(floor["minimum_weakest_family_rate"])
        and int(denominators["unsafe"]) == 0
        and int(denominators["rollback_verified"]) == attempted
    )
    walls = Counter(
        str(row["causal_wall"]) for row in evaluation["tasks"]
    )
    report = {
        "policy": "project_theseus_local_8b_qualification_disposition_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "disposition": (
            "PASS_LOCAL_8B_WORKER_COMPETENCE"
            if passed else "FAIL_LOCAL_8B_WORKER_COMPETENCE"
        ),
        "qualification_passed": passed,
        "scope": (
            "exact mlx-community/Qwen3-8B-4bit revision "
            "545dc4251c05440727734bcd94334791f6ab0192, Worker v2, "
            "three prospective repository-correctness requests, frozen "
            "decoding, tools, budgets, evaluator, and Apple M1 16GB host"
        ),
        "source_identities": {
            "evaluation_sha256": sha256_file(evaluation_path),
            "candidates_sha256": sha256_file(candidates_path),
            "freeze_sha256": sha256_file(freeze_path),
            "evaluator_manifest_sha256": sha256_file(
                evaluator_manifest_path
            ),
        },
        "frozen_floor": floor,
        "observed": {
            **denominators,
            "useful_rate": useful_rate,
            "weakest_family_useful_rate": weakest_rate,
            "family_rates": family_rates,
            "causal_wall_counts": dict(sorted(walls.items())),
        },
        "terminal_effects": {
            "matched_stack_experiments_authorized": passed,
            "original_E2_heldout_authorized": passed,
            "original_E2_heldout_remains_sealed": not passed,
            "semantic_rerun_of_consumed_cohort_authorized": False,
        },
        "causal_diagnosis": {
            "primary_wall": "LOCAL_MODEL_WORKER_EDIT_SYNTHESIS_AND_RECOVERY",
            "secondary_wall": "RETRIEVAL_PRECISION",
            "observations": [
                (
                    "Two tasks ended in explicit abstention after the model "
                    "reached the requested implementation and visible tests."
                ),
                (
                    "The only released patch changed a journal caller instead "
                    "of verify_journal and failed all four request-derived "
                    "functional assertions."
                ),
                (
                    "The GVR task emitted two identical old/new replacements; "
                    "the controller rejected both without changing the snapshot."
                ),
            ],
            "not_tested": [
                "VCM efficacy",
                "planning efficacy",
                "routing efficacy",
                "governance efficacy",
                "procedural-memory efficacy",
                "Theseus student competence",
                "other local 7-9B code models",
            ],
        },
        "next_preconditions": {
            "model_selection": (
                "Run a development-only, hardware-feasible local code-model "
                "bakeoff under matched information, tools, decoding, and total "
                "lifecycle cost; freeze the winner before a new cohort."
            ),
            "subsystem_adequacy": (
                "Qualify VCM, planning, routing, governance, and procedural "
                "reuse independently with known-good controls, negative "
                "controls, intervention fidelity, and failure injection before "
                "counting any integrated-arm failure against a subsystem bet."
            ),
            "freshness": (
                "Any successor qualification requires new request-derived "
                "tasks and hidden tests; these three tasks are consumed."
            ),
        },
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "user_facing_effects": 0,
        },
        "maximum_inference": (
            "The exact Qwen3-8B Worker v2 failed this competence gate. The "
            "result does not falsify Theseus, VCM, governance, local models "
            "generally, or any subsystem that was not intervened upon."
        ),
    }
    report["report_payload_sha256"] = stable_hash({
        key: value for key, value in report.items()
        if key not in {"created_utc", "report_payload_sha256"}
    })
    return report


def validate_inputs(
    evaluation: dict[str, Any],
    candidates: dict[str, Any],
    freeze: dict[str, Any],
    evaluation_path: Path,
    candidates_path: Path,
    evaluator_manifest_path: Path,
) -> None:
    if evaluation.get("trigger_state") != "GREEN":
        raise ValueError("evaluation_not_valid")
    if candidates.get("trigger_state") != "GREEN":
        raise ValueError("candidate_generation_not_valid")
    if freeze.get("trigger_state") != "GREEN":
        raise ValueError("freeze_not_valid")
    if evaluation.get("candidate_report_sha256") != sha256_file(
        candidates_path
    ):
        raise ValueError("evaluation_candidate_identity_mismatch")
    if evaluation.get("evaluator_manifest_sha256") != sha256_file(
        evaluator_manifest_path
    ):
        raise ValueError("evaluation_manifest_identity_mismatch")
    required = {
        "attempted",
        "released",
        "useful",
        "unsafe",
        "false_blocked",
        "rescued",
        "malformed",
        "abstained",
        "denied",
        "timed_out",
        "infrastructure_failed",
        "skipped",
        "rollback_verified",
    }
    if set(evaluation.get("denominators") or {}) != required:
        raise ValueError("complete_denominators_missing")
    if int(evaluation["denominators"]["attempted"]) != 3:
        raise ValueError("attempted_denominator_invalid")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object:{path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
