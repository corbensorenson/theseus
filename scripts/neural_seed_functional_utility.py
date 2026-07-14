#!/usr/bin/env python3
"""Freeze and execute the private functional-utility contract for neural seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neural_seed_functional_cases import ARMS, materialize_cases, public_case, stable_hash
from neural_seed_functional_verifiers import score_english_judgments, verify_candidate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/neural_seed_functional_utility.json"
DEFAULT_FREEZE = ROOT / "configs/neural_seed_functional_utility_freeze.json"
DEFAULT_MANIFEST = ROOT / "reports/private_functional_utility_manifest.json"
DEFAULT_PACKET = ROOT / "reports/private_functional_utility_candidate_packet.json"
DEFAULT_RESULT = ROOT / "reports/private_functional_utility_qualification.json"
TRAINING_SCRIPT = ROOT / "scripts/moecot_language_arm_training.py"
TRAINING_CONFIG = ROOT / "configs/moecot_language_arm_training.json"
GENERATION_WRAPPER = ROOT / "scripts/neural_seed_functional_generate.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--supersede-freeze-before-results", action="store_true")
    parser.add_argument("--supersede-reason", default="")
    parser.add_argument("--freeze-out", default=str(DEFAULT_FREEZE))
    parser.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--packet-out", default=str(DEFAULT_PACKET))
    parser.add_argument("--evaluate-candidates", default="")
    parser.add_argument("--judgments", default="")
    parser.add_argument("--out", default=str(DEFAULT_RESULT))
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = read_json(config_path)
    manifest = build_manifest(config, config_path)
    write_json(resolve(args.manifest_out), manifest)
    write_json(resolve(args.packet_out), manifest["candidate_packet"])
    if args.freeze:
        freeze_path = resolve(args.freeze_out)
        if freeze_path.exists():
            immutable = read_json(freeze_path)
            identity_gaps = validate_freeze(manifest, immutable)
            if identity_gaps and not args.supersede_freeze_before_results:
                print(json.dumps({"trigger_state": "RED", "hard_gaps": identity_gaps}, indent=2))
                return 2
            if identity_gaps:
                if not args.supersede_reason.strip():
                    raise ValueError("superseding a freeze requires --supersede-reason")
                if manifest["training_state_at_materialization"]["dense_controls_complete"]:
                    raise ValueError("cannot supersede functional contract after dense-control completion")
                freeze = build_freeze(
                    manifest,
                    config_path,
                    predecessor_sha256=stable_hash(immutable),
                    supersede_reason=args.supersede_reason.strip(),
                )
                write_json(freeze_path, freeze)
        else:
            freeze = build_freeze(manifest, config_path)
            write_json(freeze_path, freeze)
    if args.evaluate_candidates:
        freeze_path = resolve(args.freeze_out)
        if not freeze_path.is_file():
            raise ValueError("functional utility must be frozen before candidate evaluation")
        judgments = read_jsonl(resolve(args.judgments)) if args.judgments else []
        result = evaluate_bundle(
            config,
            manifest,
            read_json(resolve(args.evaluate_candidates)),
            read_json(freeze_path),
            judgments,
        )
        write_json(resolve(args.out), result)
        print(json.dumps(summary_view(result), indent=2, sort_keys=True))
        return 0 if result["trigger_state"] != "RED" else 2
    print(json.dumps(summary_view(manifest), indent=2, sort_keys=True))
    return 0 if manifest["trigger_state"] == "GREEN" else 2


def build_manifest(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    gaps = validate_config(config)
    cases = materialize_cases(config)
    expected = len(ARMS) * len(next(iter(config["arms"].values()))["families"]) * int(config["variants_per_family"])
    if len(cases) != expected:
        gaps.append("case_count_mismatch")
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids):
        gaps.append("duplicate_case_id")
    for arm in ARMS:
        if sum(case["arm_id"] == arm for case in cases) != int(config["expected_cases_per_arm"]):
            gaps.append(f"arm_case_count_mismatch:{arm}")
    overlap = source_disjoint_audit(config, cases)
    gaps.extend(overlap["hard_gaps"])
    packet_rows = [case["model_visible"] for case in cases]
    visible_keys = set(config["generator_view"])
    if any(set(row) != visible_keys for row in packet_rows):
        gaps.append("generator_packet_field_mismatch")
    forbidden = set(config["evaluator_only_fields"]) | {
        "task_family", "category", "solution", "solution_body", "tests", "hidden_tests",
        "expected", "answer", "source_task_id", "return_shape", "type_family", "required_constructs",
    }
    leaked = sorted({key for row in packet_rows for key in row if key in forbidden})
    if leaked:
        gaps.append("generator_packet_evaluator_metadata_leak:" + ",".join(leaked))
    case_contract_sha = stable_hash(
        [{key: value for key, value in case.items() if key != "model_visible"} for case in cases]
    )
    candidate_packet = {
        "policy": "project_theseus_private_functional_candidate_packet_v1",
        "contract_sha256": case_contract_sha,
        "generator_visible_fields": list(config["generator_view"]),
        "rows": packet_rows,
        "row_count": len(packet_rows),
        "evaluator_metadata_present": False,
        "public_benchmark_payload_count": 0,
    }
    training = current_training_state(config)
    toolchains = toolchain_identity()
    return {
        "policy": config["policy"],
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "mode": "frozen_contract_readiness",
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "compiler": relative(Path(__file__).resolve()),
        "compiler_sha256": sha256_file(Path(__file__).resolve()),
        "case_compiler_sha256": sha256_file(ROOT / "scripts/neural_seed_functional_cases.py"),
        "verifier_sha256": sha256_file(ROOT / "scripts/neural_seed_functional_verifiers.py"),
        "generation_wrapper_sha256": sha256_file(GENERATION_WRAPPER),
        "training_generator_sha256": sha256_file(TRAINING_SCRIPT),
        "toolchain_identity": toolchains,
        "toolchain_identity_sha256": stable_hash(toolchains),
        "v8_plan_sha256": config["v8_plan_sha256"],
        "v8_stage_signature": config["v8_stage_signature"],
        "case_contract_sha256": case_contract_sha,
        "case_count": len(cases),
        "cases_by_arm": {arm: sum(case["arm_id"] == arm for case in cases) for arm in ARMS},
        "candidate_packet": candidate_packet,
        "candidate_packet_sha256": stable_hash(candidate_packet),
        "source_disjoint_audit": overlap,
        "training_state_at_materialization": training,
        "evaluator_cases": cases,
        "hard_gaps": gaps,
        "boundaries": {
            **config["boundaries"],
            "generator_sees_verifier": False,
            "generator_sees_task_family": False,
            "public_benchmark_payload_count": 0,
            "capability_claim": "NOT_EVALUATED",
        },
    }


def build_freeze(
    manifest: dict[str, Any],
    config_path: Path,
    *,
    predecessor_sha256: str = "",
    supersede_reason: str = "",
) -> dict[str, Any]:
    if manifest["trigger_state"] != "GREEN":
        raise ValueError("cannot freeze an invalid functional contract")
    training = manifest["training_state_at_materialization"]
    return {
        "policy": "project_theseus_private_functional_utility_freeze_v1",
        "frozen_utc": now(),
        "immutable": True,
        "config": relative(config_path),
        "config_sha256": manifest["config_sha256"],
        "compiler_sha256": manifest["compiler_sha256"],
        "case_compiler_sha256": manifest["case_compiler_sha256"],
        "verifier_sha256": manifest["verifier_sha256"],
        "generation_wrapper_sha256": manifest["generation_wrapper_sha256"],
        "training_generator_sha256": manifest["training_generator_sha256"],
        "toolchain_identity_sha256": manifest["toolchain_identity_sha256"],
        "case_contract_sha256": manifest["case_contract_sha256"],
        "candidate_packet_sha256": manifest["candidate_packet_sha256"],
        "v8_plan_sha256": manifest["v8_plan_sha256"],
        "v8_stage_signature": manifest["v8_stage_signature"],
        "case_count": manifest["case_count"],
        "cases_by_arm": manifest["cases_by_arm"],
        "dense_controls_complete_at_freeze": training["dense_controls_complete"],
        "training_state_at_freeze": training,
        "evaluation_state": "NOT_EVALUATED",
        "capability_claim": "NOT_EVALUATED",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "templates_renderers_routers_tools_credit": 0,
        "supersedes_freeze_sha256": predecessor_sha256,
        "supersede_reason": supersede_reason,
    }


def source_disjoint_audit(config: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    root = resolve(config["source_disjoint"]["supervision_root"])
    prompt_hashes: dict[str, str] = {}
    normalized: dict[str, str] = {}
    target_hashes: set[str] = set()
    files = []
    rows_scanned = 0
    for split in config["source_disjoint"]["scan_splits"]:
        for path in sorted((root / split).glob("*.jsonl")):
            files.append({"path": relative(path), "sha256": sha256_file(path)})
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    rows_scanned += 1
                    prompt = str(row.get("prompt") or "")
                    prompt_hashes[hashlib.sha256(prompt.encode()).hexdigest()] = str(row.get("row_id") or "")
                    normalized[normalize_text(prompt)] = str(row.get("row_id") or "")
                    target_hash = str(row.get("target_sha256") or "")
                    if target_hash:
                        target_hashes.add(target_hash)
    gaps = []
    exact = []
    normalized_hits = []
    target_as_prompt = []
    for case in cases:
        prompt_hash = case["prompt_sha256"]
        if prompt_hash in prompt_hashes:
            exact.append({"case_id": case["case_id"], "row_id": prompt_hashes[prompt_hash]})
        norm = normalize_text(case["prompt"])
        if norm in normalized:
            normalized_hits.append({"case_id": case["case_id"], "row_id": normalized[norm]})
        if prompt_hash in target_hashes:
            target_as_prompt.append(case["case_id"])
    if exact:
        gaps.append("exact_supervision_prompt_overlap")
    if normalized_hits:
        gaps.append("normalized_supervision_prompt_overlap")
    if target_as_prompt:
        gaps.append("supervision_target_reused_as_prompt")
    return {
        "state": "GREEN" if not gaps else "RED",
        "rows_scanned": rows_scanned,
        "files": files,
        "exact_prompt_overlaps": exact,
        "normalized_prompt_overlaps": normalized_hits,
        "target_hash_as_prompt": target_as_prompt,
        "hard_gaps": gaps,
    }


def current_training_state(config: dict[str, Any]) -> dict[str, Any]:
    checkpoint_root = ROOT / "checkpoints/moecot_language_seed_v8"
    controls = {}
    for target in ("dense_total_parameter", "dense_active_parameter"):
        directory = checkpoint_root / target
        receipt = directory / "training_receipt.json"
        heartbeat = directory / "training_heartbeat.json"
        receipt_row = read_json(receipt) if receipt.is_file() else {}
        controls[target] = {
            "receipt": relative(receipt),
            "receipt_sha256": sha256_file(receipt) if receipt.is_file() else "",
            "heartbeat": relative(heartbeat),
            "heartbeat_sha256": sha256_file(heartbeat) if heartbeat.is_file() else "",
            "complete": bool(receipt_row.get("complete")),
            "optimizer_steps": int(receipt_row.get("optimizer_steps") or 0),
            "plan_sha256": receipt_row.get("plan_sha256"),
        }
    return {
        "controls": controls,
        "dense_controls_complete": all(row["complete"] for row in controls.values()),
        "functional_contract_frozen_before_control_completion": not all(row["complete"] for row in controls.values()),
    }


def evaluate_bundle(
    config: dict[str, Any], manifest: dict[str, Any], bundle: dict[str, Any], freeze: dict[str, Any], judgments: list[dict[str, Any]]
) -> dict[str, Any]:
    gaps = validate_freeze(manifest, freeze)
    cases = {case["case_id"]: case for case in manifest["evaluator_cases"]}
    rows = bundle.get("candidates") if isinstance(bundle.get("candidates"), list) else []
    ids = [str(row.get("case_id") or "") for row in rows]
    if len(ids) != len(set(ids)):
        gaps.append("duplicate_candidate_case")
    if set(ids) != set(cases):
        gaps.append("candidate_case_set_mismatch")
    provenance = audit_candidate_provenance(bundle, freeze)
    gaps.extend(provenance["hard_gaps"])
    outputs = {str(row.get("case_id") or ""): str(row.get("output") or "") for row in rows}
    verifier_rows = []
    if not gaps:
        verifier_rows = [verify_candidate(cases[case_id], outputs[case_id], config) for case_id in sorted(cases)]
    english = score_english_judgments(list(cases.values()), outputs, judgments, config) if judgments else {
        "valid": False, "faults": ["blind_english_judgments_pending"], "results": [], "quadratic_weighted_kappa": None, "passed": 0, "total": 32
    }
    code_rows = [row for row in verifier_rows if row["arm_id"] != "english"]
    by_arm = {}
    for arm in ARMS:
        arm_rows = english["results"] if arm == "english" else [row for row in code_rows if row["arm_id"] == arm]
        by_arm[arm] = metric_summary(arm_rows, expected=int(config["expected_cases_per_arm"]))
    all_complete = not gaps and english["valid"] and len(code_rows) == 128
    passed = sum(row["passed"] for row in code_rows) + int(english["passed"])
    total = len(code_rows) + (int(english["total"]) if english["valid"] else 0)
    return {
        "policy": "project_theseus_private_functional_utility_qualification_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all_complete else ("RED" if gaps else "YELLOW"),
        "evaluation_complete": all_complete,
        "freeze_sha256": stable_hash(freeze),
        "candidate_provenance": provenance,
        "summary": {
            "functional_pass_rate": passed / total if total else None,
            "passed": passed,
            "total_scored": total,
            "tail_floor": min((row["functional_pass_rate"] for row in by_arm.values() if row["functional_pass_rate"] is not None), default=None),
            "invalid_rate": sum(row.get("fault") in {"syntax_error", "markdown_fence", "candidate_too_large"} for row in code_rows) / len(code_rows) if code_rows else None,
            "timeout_rate": sum(row.get("fault") == "timeout" for row in code_rows) / len(code_rows) if code_rows else None,
            "no_output_rate": sum(row.get("fault") == "no_output" for row in code_rows) / len(code_rows) if code_rows else None,
            "english_inter_rater_agreement": english["quadratic_weighted_kappa"],
            "candidate_budget_per_case": 1,
            "pass_if_any_rate": passed / total if total else None,
            "selected_pass_rate": passed / total if total else None,
            "accepted_verified_output_per_second": (
                sum(bool(row.get("passed")) for row in code_rows)
                / (sum(float(row.get("duration_ms") or 0) for row in code_rows) / 1000.0)
                if code_rows and sum(float(row.get("duration_ms") or 0) for row in code_rows) > 0
                else None
            ),
        },
        "by_arm": by_arm,
        "english": english,
        "rows": verifier_rows,
        "hard_gaps": gaps,
        "boundaries": {
            "candidate_self_declared_flags_trusted": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "templates_renderers_routers_tools_credit": 0,
            "exact_recovery_is_functional_utility": False,
        },
    }


def audit_candidate_provenance(bundle: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    gaps = []
    if bundle.get("policy") != "project_theseus_direct_model_candidate_bundle_v1":
        gaps.append("candidate_bundle_policy_mismatch")
    if bundle.get("case_contract_sha256") != freeze["case_contract_sha256"]:
        gaps.append("candidate_bundle_contract_mismatch")
    if bundle.get("generation_function") != "moecot_language_arm_training.generate_model_text":
        gaps.append("candidate_generation_function_mismatch")
    if bundle.get("generation_wrapper_sha256") != freeze.get("generation_wrapper_sha256"):
        gaps.append("candidate_generation_wrapper_mismatch")
    if bundle.get("training_generator_sha256") != freeze.get("training_generator_sha256"):
        gaps.append("candidate_training_generator_mismatch")
    if int(bundle.get("templates_renderers_routers_tools_credit", -1)) != 0:
        gaps.append("nonzero_nonlearned_generation_credit")
    artifacts = bundle.get("checkpoint_artifacts") if isinstance(bundle.get("checkpoint_artifacts"), list) else []
    if not artifacts:
        gaps.append("checkpoint_artifacts_missing")
    for row in artifacts:
        path = resolve(str(row.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(row.get("sha256") or ""):
            gaps.append(f"checkpoint_identity_mismatch:{row.get('target_id')}")
    return {"state": "GREEN" if not gaps else "RED", "checkpoint_artifacts": artifacts, "hard_gaps": gaps}


def validate_freeze(manifest: dict[str, Any], freeze: dict[str, Any]) -> list[str]:
    gaps = []
    for key in ("config_sha256", "compiler_sha256", "case_compiler_sha256", "verifier_sha256", "generation_wrapper_sha256", "training_generator_sha256", "toolchain_identity_sha256", "case_contract_sha256", "candidate_packet_sha256", "v8_plan_sha256", "v8_stage_signature"):
        if manifest.get(key) != freeze.get(key):
            gaps.append(f"freeze_identity_mismatch:{key}")
    return gaps


def validate_config(config: dict[str, Any]) -> list[str]:
    gaps = []
    if config.get("policy") != "project_theseus_private_functional_utility_contract_v1":
        gaps.append("policy_mismatch")
    if tuple(config.get("arms", {}).keys()) != ARMS:
        gaps.append("arm_contract_mismatch")
    if "task_family" in config.get("generator_view", []):
        gaps.append("task_family_visible_to_generator")
    for key, value in config.get("boundaries", {}).items():
        if key.endswith("count") or key in {"public_training_rows_written", "external_inference_calls", "templates_renderers_routers_tools_credit"}:
            if isinstance(value, int) and value != 0:
                gaps.append(f"nonzero_boundary:{key}")
    return gaps


def metric_summary(rows: list[dict[str, Any]], *, expected: int) -> dict[str, Any]:
    return {
        "passed": sum(bool(row.get("passed")) for row in rows),
        "scored": len(rows),
        "expected": expected,
        "functional_pass_rate": sum(bool(row.get("passed")) for row in rows) / len(rows) if rows else None,
    }


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def toolchain_identity() -> dict[str, Any]:
    commands = {
        "python": [shutil.which("python3") or "/usr/bin/python3", "--version"],
        "deno": [shutil.which("deno") or "deno", "--version"],
        "cargo": [shutil.which("cargo") or "cargo", "--version"],
        "rustc": [shutil.which("rustc") or "rustc", "--version"],
        "tidy": [shutil.which("tidy") or "/usr/bin/tidy", "-v"],
        "chrome": ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--version"],
    }
    rows = {}
    for name, command in commands.items():
        executable = Path(command[0])
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10, check=False)
        rows[name] = {
            "executable": str(executable.resolve()) if executable.exists() else command[0],
            "version": completed.stdout.strip(),
            "returncode": completed.returncode,
        }
    return rows


def summary_view(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in ("policy", "created_utc", "trigger_state", "mode", "case_count", "cases_by_arm", "summary", "hard_gaps", "boundaries") if key in report}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
