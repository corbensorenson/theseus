"""Shared constants and IO helpers for the high-transfer scheduler.

This scheduler keeps benchmaxxing honest by scheduling concept pressure that
should transfer across benchmark families instead of overfitting a single public
suite. It produces diagnostic tasks only; public benchmark artifacts remain
calibration-only and are never training rows.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from high_transfer_curriculum_catalog import CurriculumCatalogContext, build_base_concepts  # noqa: E402

CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "high_transfer_curriculum_scheduler.json"
DEFAULT_MARKDOWN = REPORTS / "high_transfer_curriculum_scheduler.md"
DEFAULT_TASKS = REPORTS / "high_transfer_curriculum_tasks.jsonl"
CONVERSATION_ROTATION_SECONDS = 60 * 60
CONVERSATION_LARGE_CASE_TARGET = 72
CONVERSATION_HARD_CASE_TARGET = 96
CONVERSATION_HARD_V2_CASE_TARGET = 128
CONVERSATION_HARD_V3_CASE_TARGET = 256
CONVERSATION_HARD_V4_CASE_TARGET = 384
CONVERSATION_GRADUATION_MIN_CASES = 64
CONVERSATION_GRADUATION_ACCURACY = 0.90
GENERALIST_ROTATION_REFRESH_SECONDS = 12 * 60 * 60
REPO_REPAIR_REFRESH_SECONDS = 24 * 60 * 60
CROSS_DOMAIN_CAPSULE_REFRESH_SECONDS = 3 * 60 * 60
DECODER_SOURCE = ROOT / "crates/symliquid-cli/src/code_lm_closure.rs"
DECODER_RELEVANT_SOURCES = (
    DECODER_SOURCE,
    ROOT / "scripts/code_lm_closure.py",
    ROOT / "scripts/code_residual_curriculum.py",
    ROOT / "scripts/type_contract_diagnostic.py",
)
DECODER_FINGERPRINT_MARKERS = (
    "semantic_decoder_v2",
    "execution_shape_skeleton",
    "edge_exec_repair",
    "typed_edge_exec_receiver",
    "decoder_contract",
    "contract_guided_skeleton",
    "contract_guided_token",
    "local_adapter_edge_skeleton",
    "sts_causal_skeleton",
    "candidate_floor",
    "body_token_allowed",
    "syntax_constrained_body",
    "invalid_inline_block_header_body",
    "callable_keyword_argument",
    "archive_context_manager",
    "invalid_overcomposed_generated_line",
)
PRIVATE_PRESSURE_REPORTS = (
    REPORTS / "high_transfer_type_and_return_shape_code_residual_curriculum.json",
    REPORTS / "high_transfer_admissibility_and_interface_code_residual_curriculum.json",
    REPORTS / "high_transfer_edge_conditions_code_residual_curriculum.json",
    REPORTS / "high_transfer_algorithmic_planning_code_residual_curriculum.json",
)
PRIVATE_PRESSURE_REQUIRED_REPORTS = (
    REPORTS / "high_transfer_type_and_return_shape_code_residual_curriculum.json",
    REPORTS / "high_transfer_admissibility_and_interface_code_residual_curriculum.json",
    REPORTS / "high_transfer_edge_conditions_code_residual_curriculum.json",
    REPORTS / "high_transfer_algorithmic_planning_code_residual_curriculum.json",
)
BROAD_PRIVATE_PRESSURE_CLOSURE_REPORT = REPORTS / "code_lm_closure_private_pressure_private.json"
BROAD_PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG = "frontier_private_transfer_private_only_train_once_v1"
BROAD_PRIVATE_PRESSURE_TRAIN_ONCE_LEGACY_SLUG = "private_pressure_private_recovery_train_once_fanout_v1"
BROAD_PRIVATE_PRESSURE_TRAIN_ONCE_CLOSURE_REPORT = (
    REPORTS / f"code_lm_closure_{BROAD_PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}.json"
)
BROAD_PRIVATE_PRESSURE_TRAIN_ONCE_LEGACY_CLOSURE_REPORT = (
    REPORTS / f"code_lm_closure_{BROAD_PRIVATE_PRESSURE_TRAIN_ONCE_LEGACY_SLUG}.json"
)
BROAD_PRIVATE_PRESSURE_TRAIN_ONCE_CLOSURE_REPORTS = (
    BROAD_PRIVATE_PRESSURE_TRAIN_ONCE_CLOSURE_REPORT,
    BROAD_PRIVATE_PRESSURE_TRAIN_ONCE_LEGACY_CLOSURE_REPORT,
)
EDGE_CONTRACT_PRESSURE_REPORT = REPORTS / "high_transfer_edge_contract_4card_code_residual_curriculum.json"
EDGE_CONTRACT_CLOSURE_REPORT = REPORTS / "code_lm_closure_edge_contract_4card_private.json"
BALANCED_EDGE_CONTRACT_CONCEPT = "edge_contract_balanced_4card_private_curriculum_v2"
BALANCED_EDGE_CONTRACT_CLOSURE_CONCEPT = "edge_contract_balanced_private_closure_v2"
BALANCED_EDGE_CONTRACT_PRESSURE_REPORT = REPORTS / f"high_transfer_{BALANCED_EDGE_CONTRACT_CONCEPT}_code_residual_curriculum.json"
BALANCED_EDGE_CONTRACT_TRAIN_JSONL = Path(
    f"D:/ProjectTheseus/training_data/high_transfer/private_train/{BALANCED_EDGE_CONTRACT_CONCEPT}_residual_code_lm_tasks.jsonl"
)
BALANCED_EDGE_CONTRACT_CLOSURE_REPORT = REPORTS / "code_lm_closure_edge_contract_balanced_4card_private_v2.json"
EDGE_CASE_FULL_BODY_CONCEPT = "edge_case_full_body_private_curriculum_v1"
EDGE_CASE_FULL_BODY_CLOSURE_CONCEPT = "edge_case_full_body_private_closure_v1"
EDGE_CASE_FULL_BODY_PRESSURE_REPORT = REPORTS / f"high_transfer_{EDGE_CASE_FULL_BODY_CONCEPT}_code_residual_curriculum.json"
EDGE_CASE_FULL_BODY_TRAIN_JSONL = Path(
    f"D:/ProjectTheseus/training_data/high_transfer/private_train/{EDGE_CASE_FULL_BODY_CONCEPT}_residual_code_lm_tasks.jsonl"
)
EDGE_CASE_FULL_BODY_CLOSURE_REPORT = REPORTS / "code_lm_closure_edge_case_full_body_private_v1.json"
EDGE_CONTRACT_V2_CONCEPT = "edge_contract_v2_private_residual_curriculum"
EDGE_CONTRACT_V2_CLOSURE_CONCEPT = "edge_contract_v2_private_closure"
EDGE_CONTRACT_V2_PRESSURE_REPORT = REPORTS / f"high_transfer_{EDGE_CONTRACT_V2_CONCEPT}_code_residual_curriculum.json"
EDGE_CONTRACT_V2_TRAIN_JSONL = Path(
    f"D:/ProjectTheseus/training_data/high_transfer/private_train/{EDGE_CONTRACT_V2_CONCEPT}_residual_code_lm_tasks.jsonl"
)
EDGE_CONTRACT_V2_CLOSURE_REPORT = REPORTS / "code_lm_closure_edge_contract_v2_private.json"
EDGE_CONTRACT_V2_VERIFIER_REPORT = REPORTS / "edge_contract_v2_private_verifier.json"
CANDIDATE_FLOOR_V2_CONCEPT = "candidate_floor_v2_private_residual_curriculum"
CANDIDATE_FLOOR_V2_CLOSURE_CONCEPT = "candidate_floor_v2_private_closure"
CANDIDATE_FLOOR_V2_PRESSURE_REPORT = REPORTS / f"high_transfer_{CANDIDATE_FLOOR_V2_CONCEPT}_code_residual_curriculum.json"
CANDIDATE_FLOOR_V2_TRAIN_JSONL = Path(
    f"D:/ProjectTheseus/training_data/high_transfer/private_train/{CANDIDATE_FLOOR_V2_CONCEPT}_residual_code_lm_tasks.jsonl"
)
CANDIDATE_FLOOR_V2_CLOSURE_REPORT = REPORTS / "code_lm_closure_candidate_floor_v2_private.json"
RESIDUAL_EDGE_CASE_CONTRACT_CONCEPT = "residual_targeted_private_edge_case_contract_v1"
RESIDUAL_EDGE_CASE_CONTRACT_PRESSURE_REPORT = REPORTS / f"high_transfer_{RESIDUAL_EDGE_CASE_CONTRACT_CONCEPT}_code_residual_curriculum.json"
RESIDUAL_EDGE_CASE_CONTRACT_TRAIN_JSONL = Path(
    f"D:/ProjectTheseus/training_data/high_transfer/private_train/{RESIDUAL_EDGE_CASE_CONTRACT_CONCEPT}_residual_code_lm_tasks.jsonl"
)
DECODER_V2_PRIVATE_ABLATION_REPORT = REPORTS / "decoder_v2_private_ablation_gate.json"
PRIVATE_TYPE_SHAPE_RECEIVER_ABLATION_REPORT = REPORTS / "private_type_shape_receiver_ablation.json"
EDGE_OBLIGATION_PRIVATE_PRESSURE_REPORT = REPORTS / "edge_obligation_decode_gate_v1_private_pressure_private.json"
DECODER_PLAN_IR_REPORT = REPORTS / "decoder_plan_ir_private_pressure.json"
DECODER_PLAN_IR_CODE_LM_ADAPTER_REPORT = REPORTS / "decoder_plan_ir_code_lm_adapter.json"
DECODER_PLAN_IR_CODE_LM_TRAIN_JSONL = Path(
    "D:/ProjectTheseus/training_data/decoder_plan_ir/private_train/decoder_plan_ir_code_lm_rows.jsonl"
)
PRIVATE_PRESSURE_REPORTS = (
    *PRIVATE_PRESSURE_REPORTS,
    BALANCED_EDGE_CONTRACT_PRESSURE_REPORT,
    EDGE_CASE_FULL_BODY_PRESSURE_REPORT,
    EDGE_CONTRACT_V2_PRESSURE_REPORT,
    RESIDUAL_EDGE_CASE_CONTRACT_PRESSURE_REPORT,
    DECODER_PLAN_IR_CODE_LM_ADAPTER_REPORT,
)
PRIVATE_PRESSURE_REQUIRED_REPORTS = (
    *PRIVATE_PRESSURE_REQUIRED_REPORTS,
    BALANCED_EDGE_CONTRACT_PRESSURE_REPORT,
    RESIDUAL_EDGE_CASE_CONTRACT_PRESSURE_REPORT,
    DECODER_PLAN_IR_CODE_LM_ADAPTER_REPORT,
)
PRIVATE_PRESSURE_DIAGNOSTIC_ONLY_REPORTS = (
    REPORTS / "high_transfer_typed_interface_skeleton_code_residual_curriculum.json",
    CANDIDATE_FLOOR_V2_PRESSURE_REPORT,
)
RECEIVER_RECALIBRATION_PRESSURE_REPORTS = (*PRIVATE_PRESSURE_REPORTS, EDGE_CONTRACT_PRESSURE_REPORT)

CONCEPTS = build_base_concepts(
    CurriculumCatalogContext(
        root=ROOT,
        balanced_edge_contract_concept=BALANCED_EDGE_CONTRACT_CONCEPT,
        balanced_edge_contract_closure_concept=BALANCED_EDGE_CONTRACT_CLOSURE_CONCEPT,
        balanced_edge_contract_train_jsonl=BALANCED_EDGE_CONTRACT_TRAIN_JSONL,
        balanced_edge_contract_pressure_report=BALANCED_EDGE_CONTRACT_PRESSURE_REPORT,
        edge_case_full_body_concept=EDGE_CASE_FULL_BODY_CONCEPT,
        edge_case_full_body_closure_concept=EDGE_CASE_FULL_BODY_CLOSURE_CONCEPT,
        edge_case_full_body_train_jsonl=EDGE_CASE_FULL_BODY_TRAIN_JSONL,
        edge_case_full_body_pressure_report=EDGE_CASE_FULL_BODY_PRESSURE_REPORT,
        edge_contract_v2_concept=EDGE_CONTRACT_V2_CONCEPT,
        edge_contract_v2_closure_concept=EDGE_CONTRACT_V2_CLOSURE_CONCEPT,
        edge_contract_v2_train_jsonl=EDGE_CONTRACT_V2_TRAIN_JSONL,
        edge_contract_v2_pressure_report=EDGE_CONTRACT_V2_PRESSURE_REPORT,
        candidate_floor_v2_concept=CANDIDATE_FLOOR_V2_CONCEPT,
        candidate_floor_v2_closure_concept=CANDIDATE_FLOOR_V2_CLOSURE_CONCEPT,
        candidate_floor_v2_train_jsonl=CANDIDATE_FLOOR_V2_TRAIN_JSONL,
        candidate_floor_v2_pressure_report=CANDIDATE_FLOOR_V2_PRESSURE_REPORT,
        residual_edge_case_contract_concept=RESIDUAL_EDGE_CASE_CONTRACT_CONCEPT,
        residual_edge_case_contract_train_jsonl=RESIDUAL_EDGE_CASE_CONTRACT_TRAIN_JSONL,
        residual_edge_case_contract_pressure_report=RESIDUAL_EDGE_CASE_CONTRACT_PRESSURE_REPORT,
        conversation_large_case_target=CONVERSATION_LARGE_CASE_TARGET,
        conversation_hard_case_target=CONVERSATION_HARD_CASE_TARGET,
        conversation_hard_v2_case_target=CONVERSATION_HARD_V2_CASE_TARGET,
        conversation_hard_v3_case_target=CONVERSATION_HARD_V3_CASE_TARGET,
        conversation_hard_v4_case_target=CONVERSATION_HARD_V4_CASE_TARGET,
    )
)



def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def stable_id(*parts: Any) -> str:
    digest = hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"transfer_{digest}"


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        return rows
    except (OSError, json.JSONDecodeError):
        return []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
