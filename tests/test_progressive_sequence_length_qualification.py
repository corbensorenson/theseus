from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mtp_matched_adequacy as corpus
import progressive_sequence_length_qualification as progressive


def sample_row() -> dict[str, object]:
    return {
        "arm_id": "python",
        "row_id": "row",
        "source_identity": "source",
        "dataset_id": "private",
        "license_spdx": "MIT",
        "sequence": list(range(401)),
        "target_mask_start": 100,
        "target_byte_count": 300,
    }


def test_config_precommits_exact_128_256_512_curriculum() -> None:
    config = progressive.load_config(
        ROOT / "configs/progressive_sequence_length_qualification.json"
    )

    assert [
        row["maximum_sequence_tokens"] for row in config["curriculum"]
    ] == [128, 256, 512]
    assert config["curriculum"][-1]["stop_step"] == 128
    assert config["hard_boundaries"]["supervised_token_dropping"] is False


def test_windowing_preserves_every_supervised_target_position_once() -> None:
    row = sample_row()
    expected = progressive.target_position_count(row)

    for width in (128, 256, 512):
        windows = progressive.window_supervised_row(
            row, width, context_fraction=0.5
        )
        assert all(len(window["sequence"]) - 1 <= width for window in windows)
        assert sum(
            progressive.target_position_count(window) for window in windows
        ) == expected
        x, y, mask = corpus.make_batch(windows, width)
        assert int(mask.sum()) == expected
        assert x.shape == y.shape == mask.shape


def test_padding_does_not_change_supervision_or_values() -> None:
    windows = progressive.window_supervised_row(
        sample_row(), 128, context_fraction=0.5
    )
    padded = progressive.pad_rows(windows, len(windows) + 3)
    x, y, mask = corpus.make_batch(padded, 128)

    assert len(padded) == len(windows) + 3
    assert int(mask.sum()) == progressive.target_position_count(sample_row())
    assert np.count_nonzero(mask[-3:]) == 0
    assert np.count_nonzero(x[-3:]) == 0
    assert np.count_nonzero(y[-3:]) == 0


def test_curriculum_boundaries_are_unambiguous() -> None:
    config = progressive.load_config(
        ROOT / "configs/progressive_sequence_length_qualification.json"
    )

    assert progressive.curriculum_width(config, 1) == 128
    assert progressive.curriculum_width(config, 48) == 128
    assert progressive.curriculum_width(config, 49) == 256
    assert progressive.curriculum_width(config, 88) == 256
    assert progressive.curriculum_width(config, 89) == 512
    assert progressive.curriculum_width(config, 128) == 512
