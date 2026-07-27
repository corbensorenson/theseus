from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/exact_decoder_block_mlx_control.py"
SPEC = importlib.util.spec_from_file_location(
    "exact_decoder_block_mlx_control", PATH
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_frozen_control_values_match_exact_block_authority() -> None:
    parameters, hidden, target, mask = module.frozen_values()
    assert sum(value.size for value in parameters) == 3_015_680
    assert hidden.shape == (128, 512)
    assert target.shape == (128, 512)
    assert mask.sum() == 125
    assert all(value.dtype == np.float32 for value in parameters)


def test_attention_boundary_values_are_half_published() -> None:
    parameters, hidden, _, _ = module.frozen_values()
    for value in (parameters[0], parameters[1], parameters[2], parameters[3], hidden):
        assert np.array_equal(value, value.astype(np.float16).astype(np.float32))
