from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/exact_decoder_block_qualification.py"
SPEC = importlib.util.spec_from_file_location("exact_decoder_block_qualification", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def config() -> dict:
    return module.load_config(ROOT / "configs/exact_decoder_block_qualification.json")


def test_exact_block_schema_covers_all_nine_parameter_leaves() -> None:
    cfg = config()
    schema = module.canonical_parameter_schema(cfg)
    assert list(schema) == [
        "attention_norm.weight",
        "attention.q_proj.weight",
        "attention.k_proj.weight",
        "attention.v_proj.weight",
        "attention.out_proj.weight",
        "ffn_norm.weight",
        "feed_forward.gate.weight",
        "feed_forward.up.weight",
        "feed_forward.down.weight",
    ]
    assert module.parameter_count(cfg) == 3_015_680


def test_gqa_mapping_is_contiguous_not_alternating() -> None:
    owners = [module.contiguous_gqa_owner(head, 8, 2) for head in range(8)]
    assert owners == [0, 0, 0, 0, 1, 1, 1, 1]
    assert owners != [0, 1, 0, 1, 0, 1, 0, 1]


def test_split_half_rotation_is_not_adjacent_pair_rotation() -> None:
    value = np.arange(8, dtype=np.float32)
    rotated = module.split_half_rotate_numpy(value)
    np.testing.assert_array_equal(rotated, [-4, -5, -6, -7, 0, 1, 2, 3])
    assert not np.array_equal(rotated, [-1, 0, -3, 2, -5, 4, -7, 6])


def test_schema_fails_closed_on_shape_drift() -> None:
    cfg = copy.deepcopy(config())
    cfg["parameter_schema"]["attention.k_proj.weight"] = [512, 512]
    with pytest.raises(module.QualificationFault, match="parameter_schema_mismatch"):
        module.validate_parameter_schema(cfg)


def test_query_head_divisibility_is_required() -> None:
    with pytest.raises(
        module.QualificationFault, match="query_heads_not_divisible_by_kv_heads"
    ):
        module.contiguous_gqa_owner(0, 8, 3)
