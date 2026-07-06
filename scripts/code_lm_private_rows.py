"""Canonical private-row inputs for Code LM training envelopes."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def training_data_root() -> Path:
    configured = os.environ.get("THESEUS_TRAINING_DATA_ROOT", "").strip()
    if configured:
        return Path(configured)
    if sys.platform.startswith("win"):
        return Path("D:/ProjectTheseus/training_data")
    return ROOT / "data" / "training_data"


def training_data_path(*parts: str) -> str:
    return str(training_data_root().joinpath(*parts)).replace("\\", "/")


DEFAULT_EXTRA_PRIVATE_TRAIN_JSONL = (
    training_data_path("open_code_pantry", "private_train", "open_code_expressions.jsonl")
)
DEFAULT_RESIDUAL_PRIVATE_TRAIN_JSONL = (
    training_data_path("residual_code_curriculum", "private_train", "residual_code_lm_tasks.jsonl")
)
DEFAULT_REPO_REPAIR_PRIVATE_TRAIN_JSONL = (
    training_data_path("long_horizon_programming", "private_train", "repo_repair_code_lm_rows.jsonl")
)
DEFAULT_NO_ADMISSIBLE_REPAIR_POLICY_JSONL = training_data_path(
    "candidate_coverage", "private_train", "no_admissible_repair_policy_rows.jsonl"
)
STS_DEFAULT_RESIDUAL_PRESSURE_PRIVATE_ROWS = (
    training_data_path(
        "high_transfer",
        "private_train",
        "sts_default_algorithm_choice_local_adapter_residual_code_lm_tasks.jsonl",
    ),
)

BASE_PROMOTION_SAFE_HIGH_TRANSFER_PRIVATE_ROWS = (
    training_data_path(
        "high_transfer",
        "private_train",
        "broad_private_generalization_ladder_v1_code_lm_tasks.jsonl",
    ),
    training_data_path("high_transfer", "private_train", "private_residual_repair_v3_code_lm_tasks.jsonl"),
    training_data_path(
        "high_transfer",
        "private_train",
        "edge_contract_v3_verifier_mismatch_public_transfer_private_curriculum_code_lm_tasks.jsonl",
    ),
    training_data_path(
        "high_transfer",
        "private_train",
        "targeted_private_residual_curriculum_v2_residual_code_lm_tasks.jsonl",
    ),
    training_data_path("high_transfer", "private_train", "type_and_return_shape_residual_code_lm_tasks.jsonl"),
    training_data_path(
        "high_transfer",
        "private_train",
        "return_type_shape_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl",
    ),
    training_data_path("high_transfer", "private_train", "type_contract_decoder_feedback.jsonl"),
    training_data_path("high_transfer", "private_train", "admissibility_and_interface_residual_code_lm_tasks.jsonl"),
    training_data_path("high_transfer", "private_train", "edge_conditions_residual_code_lm_tasks.jsonl"),
    training_data_path("high_transfer", "private_train", "edge_contract_4card_residual_code_lm_tasks.jsonl"),
    training_data_path(
        "high_transfer",
        "private_train",
        "edge_contract_balanced_4card_private_curriculum_v2_residual_code_lm_tasks.jsonl",
    ),
    training_data_path("high_transfer", "private_train", "edge_case_full_body_private_curriculum_v1_residual_code_lm_tasks.jsonl"),
    training_data_path("high_transfer", "private_train", "edge_contract_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl"),
    training_data_path(
        "high_transfer",
        "private_train",
        "residual_targeted_private_edge_case_contract_v1_residual_code_lm_tasks.jsonl",
    ),
    training_data_path(
        "high_transfer",
        "private_train",
        "candidate_floor_adapter_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl",
    ),
    training_data_path(
        "high_transfer",
        "private_train",
        "parsing_encoding_v1_private_residual_curriculum_residual_code_lm_tasks.jsonl",
    ),
    training_data_path("high_transfer", "private_train", "algorithmic_planning_residual_code_lm_tasks.jsonl"),
    training_data_path("high_transfer", "private_train", "execution_shaped_programs_residual_code_lm_tasks.jsonl"),
    training_data_path("decoder_plan_ir", "private_train", "decoder_plan_ir_code_lm_rows.jsonl"),
)

BROAD_FLOOR_RECOVERY_PRIVATE_ROWS = (
    training_data_path("high_transfer", "private_train", "broad_public_transfer_floor_ratchet_v2_private_rows.jsonl"),
    training_data_path("high_transfer", "private_train", "broad_floor_v3_public_residual_private_recovery_rows.jsonl"),
    training_data_path(
        "high_transfer",
        "private_train",
        "broad_public_code_transfer_floor_recovery_v1_residual_code_lm_tasks.jsonl",
    ),
    DEFAULT_RESIDUAL_PRIVATE_TRAIN_JSONL,
)

DIAGNOSTIC_ONLY_HIGH_TRANSFER_PRIVATE_ROWS = (
    training_data_path("high_transfer", "private_train", "typed_interface_skeleton_residual_code_lm_tasks.jsonl"),
    training_data_path(
        "high_transfer",
        "private_train",
        "candidate_floor_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl",
    ),
)

REQUIRED_PROMOTION_SAFE_HIGH_TRANSFER_PRIVATE_FILES = {
    training_data_path(
        "high_transfer",
        "private_train",
        "residual_targeted_private_edge_case_contract_v1_residual_code_lm_tasks.jsonl",
    ),
    training_data_path("decoder_plan_ir", "private_train", "decoder_plan_ir_code_lm_rows.jsonl"),
}


def existing_optional_private_rows(rows: tuple[str, ...]) -> list[str]:
    return [path for path in rows if Path(path).exists()]


def existing_no_admissible_repair_policy_rows() -> list[Path]:
    root = training_data_root()
    candidates = [Path(DEFAULT_NO_ADMISSIBLE_REPAIR_POLICY_JSONL)]
    high_transfer_dir = root / "high_transfer" / "private_train"
    if high_transfer_dir.exists():
        candidates.extend(sorted(high_transfer_dir.glob("no_admissible_repair_policy_rows*.jsonl")))
    return [path for path in candidates if path.exists()]


def default_no_admissible_repair_policy_jsonl() -> str:
    configured = os.environ.get("THESEUS_NO_ADMISSIBLE_REPAIR_POLICY_JSONL", "").strip()
    if configured:
        return configured
    existing = existing_no_admissible_repair_policy_rows()
    if existing:
        newest = max(existing, key=lambda path: path.stat().st_mtime)
        return str(newest).replace("\\", "/")
    return DEFAULT_NO_ADMISSIBLE_REPAIR_POLICY_JSONL


def high_transfer_private_rows_string(
    *,
    include_broad_floor_recovery: bool = False,
    include_sts_default_residual_pressure: bool = True,
) -> str:
    rows = []
    if include_broad_floor_recovery:
        rows.extend(BROAD_FLOOR_RECOVERY_PRIVATE_ROWS)
    rows.extend(BASE_PROMOTION_SAFE_HIGH_TRANSFER_PRIVATE_ROWS)
    if include_sts_default_residual_pressure:
        rows.extend(existing_optional_private_rows(STS_DEFAULT_RESIDUAL_PRESSURE_PRIVATE_ROWS))
    return ";".join(rows)
