#!/usr/bin/env python3
"""Prove governed teacher-distillation admission without polluting real ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REAL_MANIFEST = REPORTS / "teacher_distillation_manifest.json"
REAL_LEDGER = REPORTS / "teacher_distillation_ledger.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/teacher_distillation_policy.json")
    parser.add_argument("--out", default="reports/teacher_distillation_admission_smoke.json")
    parser.add_argument("--markdown-out", default="reports/teacher_distillation_admission_smoke.md")
    args = parser.parse_args()

    before = path_hashes([REAL_MANIFEST, REAL_LEDGER])
    source_policy = read_json(ROOT / args.policy)
    with tempfile.TemporaryDirectory(prefix="theseus_teacher_distillation_smoke_") as temp_name:
        temp = Path(temp_name)
        teacher_calls = temp / "teacher_calls.jsonl"
        policy_path = temp / "teacher_distillation_policy.json"
        manifest_path = temp / "teacher_distillation_manifest.json"
        ledger_path = temp / "teacher_distillation_ledger.jsonl"
        audit_path = temp / "teacher_distillation_manifest_audit.json"
        audit_md = temp / "teacher_distillation_manifest_audit.md"
        gate_path = temp / "teacher_distillation_gate.json"
        gate_md = temp / "teacher_distillation_gate.md"

        fixture_call = build_fixture_call()
        write_jsonl(teacher_calls, [fixture_call])
        temp_policy = make_temp_policy(source_policy, manifest_path, ledger_path)
        write_json(policy_path, temp_policy)

        builder_proc = run(
            [
                sys.executable,
                "scripts/teacher_distillation_manifest_builder.py",
                "--policy",
                str(policy_path),
                "--teacher-calls",
                str(teacher_calls),
                "--manifest-out",
                str(manifest_path),
                "--ledger-out",
                str(ledger_path),
                "--audit-out",
                str(audit_path),
                "--markdown-out",
                str(audit_md),
            ]
        )
        teacher_cap = float(temp_policy["teacher_share"]["max_initial_training_ratio"])
        self_row_count = max(1, math.ceil(1.0 / teacher_cap) - 1)
        append_verified_self_rows(ledger_path, count=self_row_count)
        gate_proc = run(
            [
                sys.executable,
                "scripts/teacher_distillation_gate.py",
                "--policy",
                str(policy_path),
                "--out",
                str(gate_path),
                "--markdown-out",
                str(gate_md),
            ]
        )
        manifest = read_json(manifest_path)
        audit = read_json(audit_path)
        gate = read_json(gate_path)
        temp_artifact_hashes = path_hashes([teacher_calls, policy_path, manifest_path, ledger_path, audit_path, gate_path])

    after = path_hashes([REAL_MANIFEST, REAL_LEDGER])
    manifest_summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    gate_summary = gate.get("summary") if isinstance(gate.get("summary"), dict) else {}
    payload = {
        "policy": "project_theseus_teacher_distillation_admission_smoke_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if smoke_passed(builder_proc, gate_proc, manifest, gate, before, after) else "YELLOW",
        "summary": {
            "builder_returncode": builder_proc["returncode"],
            "gate_returncode": gate_proc["returncode"],
            "manifest_trigger_state": audit.get("trigger_state"),
            "gate_trigger_state": gate.get("trigger_state"),
            "gate_distillation_allowed": gate.get("distillation_allowed"),
            "admitted_manifest_rows": manifest_summary.get("row_count"),
            "manifest_verifier_pass_rate": manifest_summary.get("verifier_pass_rate"),
            "manifest_public_overlap_hits": manifest_summary.get("public_overlap_hits"),
            "manifest_holdout_overlap_hits": manifest_summary.get("holdout_overlap_hits"),
            "teacher_share": gate.get("teacher_share"),
            "teacher_share_within_cap": gate_summary.get("teacher_share_within_cap"),
            "verified_self_rows_added_to_temp_ledger": self_row_count,
            "real_teacher_manifest_unchanged": before.get(str(REAL_MANIFEST)) == after.get(str(REAL_MANIFEST)),
            "real_teacher_ledger_unchanged": before.get(str(REAL_LEDGER)) == after.get(str(REAL_LEDGER)),
            "real_training_rows_written": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
        },
        "fixture_semantics": (
            "This is a local temporary fixture that proves admission mechanics only. "
            "It is not a real teacher training row and is not written to the real manifest, ledger, or training data."
        ),
        "real_artifact_hashes_before": before,
        "real_artifact_hashes_after": after,
        "temp_artifact_hashes": temp_artifact_hashes,
        "builder": builder_proc,
        "gate": gate_proc,
        "next_action": next_action(manifest, gate, before, after),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, payload)
    write_text(ROOT / args.markdown_out, render_markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["trigger_state"] == "GREEN" else 1


def build_fixture_call() -> dict[str, Any]:
    input_text = (
        "Private distillation fixture: decide whether a retained teacher row may be routed "
        "to training when runtime external serving is forbidden and all local admission checks pass."
    )
    code_lm_task = {
        "task_id": "teacher_smoke_private_sum_pair_v1",
        "split": "train",
        "category": "teacher_smoke_private_sum_pair",
        "concept_residual_label": "teacher_smoke_private_sum_pair",
        "prompt": "Return the sum of two integer inputs.",
        "entry_point": "teacher_smoke_private_sum_pair_v1",
        "solution_body": "return data + other",
        "tests": "assert teacher_smoke_private_sum_pair_v1(2, 3) == 5\nassert teacher_smoke_private_sum_pair_v1(-4, 1) == -3\n",
        "decoder_contract": {"visible_arg_count_hint": 2},
        "public_benchmark": False,
        "public_prompt": False,
    }
    target_text = json.dumps({"code_lm_task": code_lm_task}, sort_keys=True)
    candidate = {
        "row_id": "smoke_private_teacher_distillation_admission_v0",
        "source_kind": "teacher_distillation",
        "task_family": "teacher_distillation_admission_boundary",
        "input_text": input_text,
        "target_text": target_text,
        "target_hash": sha256_text(target_text),
        "code_lm_task": code_lm_task,
        "license_spdx": "project-internal",
        "provenance": {
            "kind": "local_smoke_fixture",
            "script": "scripts/teacher_distillation_admission_smoke.py",
            "public_eval_payload": False,
        },
        "public_benchmark": False,
        "public_prompt": False,
        "public_overlap_hits": 0,
        "holdout_overlap_hits": 0,
        "runtime_serving": "forbidden",
        "admission_checks": {
            "provenance_retained": True,
            "license_checked": True,
            "leakage_audited": True,
            "verifier_accepted": True,
            "runtime_serving_forbidden": True,
            "public_benchmark_excluded": True,
        },
        "notes": "Smoke-only row; proves the manifest/gate path and is never written to real training data.",
    }
    response_json = {
        "reason_for_call": "architecture_wall",
        "wall_type": "teacher_distillation_path_unproven",
        "blocked_gate": "teacher_distillation_manifest_has_rows",
        "residual_family": "governed_training_row_admission",
        "local_evidence_used": ["temp smoke fixture"],
        "diagnosis": "The distillation path should admit rows only when all required local checks are present.",
        "recommended_intervention": "Admit the temporary fixture row into a temporary manifest and gate it with a self-row denominator.",
        "implementation_plan": ["Build temporary teacher_calls.jsonl", "Run manifest builder", "Run gate"],
        "local_executor_inputs": ["temporary fixture only"],
        "private_eval_plan": ["Check temporary manifest row_count=1 and verifier pass rate=1.0"],
        "public_calibration_plan": [],
        "verification_steps": ["Verify real manifest and ledger hashes are unchanged"],
        "promotion_gates": ["No real teacher rows admitted by this smoke"],
        "promotion_rollback_rule": "Discard temporary artifacts after summary is written.",
        "risks": ["Mistaking a smoke fixture for real training data"],
        "evidence_gaps": [],
        "anti_goals_acknowledged": ["No public benchmark answers", "No runtime external serving"],
        "forbidden_actions_acknowledged": ["No public solutions in training", "No teacher apply mode", "No wrappers/templates"],
        "experiment_spec": None,
        "confidence": "high",
        "distill_into_local_rules": True,
        "distillation_training_row": candidate,
    }
    response_text = json.dumps(response_json, sort_keys=True)
    return {
        "request_id": "teacher_distillation_smoke_fixture",
        "created_utc": now(),
        "completed_utc": now(),
        "provider": "openai",
        "model": "codex-smoke-fixture",
        "reason_for_call": "architecture_wall",
        "mode": "distillation",
        "status": "completed",
        "prompt_sha256": sha256_text("local smoke fixture"),
        "response_text": response_text,
        "response_json": response_json,
        "external_inference_calls": 0,
    }


def make_temp_policy(source_policy: dict[str, Any], manifest_path: Path, ledger_path: Path) -> dict[str, Any]:
    policy = json.loads(json.dumps(source_policy))
    policy["manifest_path"] = str(manifest_path)
    policy["ledger_path"] = str(ledger_path)
    policy["operator_unlock_flag"] = str(manifest_path.parent / "operator_unlock_not_required.flag")
    policy["default_state"] = "governed_training_enabled"
    policy.setdefault("neural_seed", {})["required"] = False
    cap = float(policy.get("teacher_share", {}).get("max_initial_training_ratio", 0.1) or 0.1)
    if not 0.0 < cap <= 0.1:
        raise ValueError("teacher distillation smoke requires the configured sparse teacher cap")
    return policy


def append_verified_self_rows(path: Path, *, count: int) -> None:
    rows = []
    for index in range(count):
        rows.append(
            {
                "ledger_event_id": f"smoke_verified_self_generated_{index}",
                "created_utc": now(),
                "source_kind": "verified_self_generated",
                "accepted": True,
                "training_admission_status": "accepted_by_manifest_pending_gate",
                "runtime_serving": "local_only",
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "smoke_fixture": True,
            }
        )
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def smoke_passed(
    builder_proc: dict[str, Any],
    gate_proc: dict[str, Any],
    manifest: dict[str, Any],
    gate: dict[str, Any],
    before: dict[str, str],
    after: dict[str, str],
) -> bool:
    manifest_summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    return bool(
        builder_proc.get("returncode") == 0
        and gate_proc.get("returncode") == 0
        and manifest_summary.get("row_count") == 1
        and float(manifest_summary.get("verifier_pass_rate") or 0.0) >= 0.95
        and gate.get("trigger_state") == "GREEN"
        and gate.get("distillation_allowed") is True
        and before.get(str(REAL_MANIFEST)) == after.get(str(REAL_MANIFEST))
        and before.get(str(REAL_LEDGER)) == after.get(str(REAL_LEDGER))
    )


def next_action(manifest: dict[str, Any], gate: dict[str, Any], before: dict[str, str], after: dict[str, str]) -> str:
    if before.get(str(REAL_MANIFEST)) != after.get(str(REAL_MANIFEST)) or before.get(str(REAL_LEDGER)) != after.get(str(REAL_LEDGER)):
        return "Investigate immediately: smoke mutated the real teacher manifest or ledger."
    if gate.get("trigger_state") != "GREEN":
        return "Fix the admission/gate mechanics before requesting live distillation rows."
    row_count = (manifest.get("summary") or {}).get("row_count")
    return (
        f"Admission mechanics are proven with {row_count} temporary row. "
        "Next live step is a governed distillation teacher call only when local evidence contains a verifier-accepted private row shape."
    )


def run(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def path_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    return "\n".join(
        [
            "# Teacher Distillation Admission Smoke",
            "",
            f"- trigger_state: `{payload.get('trigger_state')}`",
            f"- manifest_trigger_state: `{summary.get('manifest_trigger_state')}`",
            f"- gate_trigger_state: `{summary.get('gate_trigger_state')}`",
            f"- admitted_manifest_rows: `{summary.get('admitted_manifest_rows')}`",
            f"- teacher_share_within_cap: `{summary.get('teacher_share_within_cap')}`",
            f"- real_teacher_manifest_unchanged: `{summary.get('real_teacher_manifest_unchanged')}`",
            f"- real_teacher_ledger_unchanged: `{summary.get('real_teacher_ledger_unchanged')}`",
            f"- external_inference_calls: `{summary.get('external_inference_calls')}`",
            f"- next_action: {payload.get('next_action')}",
            "",
        ]
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
