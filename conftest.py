from __future__ import annotations

import ast
import inspect
import json
import os
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
FREEZE_CONTRACT = ROOT / "configs" / "pretraining_architecture_freeze.json"


def _accelerator_selectors() -> tuple[set[str], set[str]]:
    config = json.loads(FREEZE_CONTRACT.read_text(encoding="utf-8"))
    files: set[str] = set()
    accelerator = config.get("accelerator_replay") or {}
    nodeids = {str(value) for value in accelerator.get("guarded_test_nodeids") or []}
    for shard in accelerator.get("shards") or []:
        for token in shard.get("command") or []:
            value = str(token)
            if not value.startswith("tests/"):
                continue
            if "::" in value:
                nodeids.add(value)
            else:
                files.add(value)
    return files, nodeids


def _source_initializes_accelerator(item: pytest.Item) -> bool:
    try:
        source = inspect.getsource(item.obj)
    except (AttributeError, OSError, TypeError):
        return False
    if "import mlx" in source or "from mlx" in source:
        return True
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "importorskip"
        ):
            continue
        module = node.args[0]
        if (
            isinstance(module, ast.Constant)
            and isinstance(module.value, str)
            and (module.value == "mlx" or module.value.startswith("mlx."))
        ):
            return True
    return False


def _has_top_level_accelerator_import(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return False
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        if any(name == "mlx" or name.startswith("mlx.") for name in names):
            return True
    return False


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    del config
    if os.environ.get("THESEUS_GUARDED_ACCELERATOR_CHILD") == "1":
        return None
    path = Path(str(collection_path)).resolve()
    if path.suffix != ".py" or ROOT not in path.parents:
        return None
    files, _nodeids = _accelerator_selectors()
    relative_file = str(path.relative_to(ROOT))
    if relative_file in files or _has_top_level_accelerator_import(path):
        return True
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    files, nodeids = _accelerator_selectors()
    marker = pytest.mark.accelerator
    for item in items:
        relative_file = str(Path(str(item.path)).resolve().relative_to(ROOT))
        if (
            relative_file in files
            or item.nodeid in nodeids
            or _source_initializes_accelerator(item)
        ):
            item.add_marker(marker)


@pytest.fixture(autouse=True)
def _deny_ambient_accelerator_execution(request: pytest.FixtureRequest) -> None:
    if (
        request.node.get_closest_marker("accelerator") is not None
        and os.environ.get("THESEUS_GUARDED_ACCELERATOR_CHILD") != "1"
    ):
        pytest.skip("accelerator test requires scripts/host_resource_safety.py")
