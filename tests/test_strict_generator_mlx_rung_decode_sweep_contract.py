#!/usr/bin/env python3
"""Keep rung replay synchronized with the canonical strict decode contract."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from strict_generator_mlx_decode_eval import run_decode_eval  # noqa: E402


def test_rung_sweep_forwards_every_required_decode_option() -> None:
    source_path = SCRIPTS / "strict_generator_mlx_rung_decode_sweep.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_decode_eval"
    ]
    assert len(calls) == 1
    forwarded = {keyword.arg for keyword in calls[0].keywords if keyword.arg}
    required = {
        name
        for name, parameter in inspect.signature(run_decode_eval).parameters.items()
        if name != "config"
        and parameter.kind == inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
    }
    assert required <= forwarded, f"rung sweep is missing decode options: {sorted(required - forwarded)}"
