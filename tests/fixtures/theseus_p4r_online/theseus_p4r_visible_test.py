#!/usr/bin/env python3
"""Bounded visible feedback for the three new P4R tasks."""

from __future__ import annotations

import ast
import copy
import pickle as stdlib_pickle
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4


class Strip(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.returns = None
        node.decorator_list = []
        return node


def function(path: str, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    node = Strip().visit(copy.deepcopy(matches[0]))
    ast.fix_missing_locations(module := ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def method(path: str, owner_name: str, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner_name)
    node = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node = Strip().visit(copy.deepcopy(node))
    ast.fix_missing_locations(module := ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def black() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_file = Path(tmp) / "cache.pickle"
        cache_file.write_bytes(b"cache")
        fake_pickle = SimpleNamespace(
            load=lambda _handle: (_ for _ in ()).throw(PermissionError("denied")),
            UnpicklingError=stdlib_pickle.UnpicklingError,
        )
        read = method(
            "src/black/cache.py", "Cache", "read",
            {"get_cache_file": lambda _mode: cache_file, "pickle": fake_pickle, "FileData": tuple},
        )

        class Cache:
            def __init__(self, mode: Any, path: Path, file_data: dict[str, Any] | None = None) -> None:
                self.file_data = file_data or {}

        Cache.read = classmethod(read)
        assert Cache.read("mode").file_data == {}, "P4R_VISIBLE_black_PRIMARY"


def django() -> None:
    path = "django/urls/utils.py"
    group = function(path, "_get_group_start_end", {})
    groups = function(path, "_find_groups", {"_get_group_start_end": group})
    replace = function(path, "replace_unnamed_groups", {
        "_find_groups": groups, "_UNNAMED_GROUP_MATCHER": re.compile(r"\(")
    })
    assert replace(r"^(\w+)/b/(\w+)$") == r"^<var>/b/<var>$", "P4R_VISIBLE_django_PRIMARY"


def celery() -> None:
    def key_t(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode()
        raise TypeError(type(value).__name__)

    method_under_test = method(
        "celery/backends/base.py", "BaseKeyValueStoreBackend", "_get_key_for", {"UUID": UUID}
    )
    owner = SimpleNamespace(key_t=key_t)
    identifier = uuid4()
    assert method_under_test(owner, b"task-", identifier) == method_under_test(
        owner, b"task-", str(identifier)
    ), "P4R_VISIBLE_celery_PRIMARY"


CASES = {"black": black, "django": django, "celery": celery}


if __name__ == "__main__":
    selected = sys.argv[1]
    try:
        CASES[selected]()
    except Exception:
        print(f"P4R_VISIBLE_{selected}_PRIMARY", file=sys.stderr)
        raise
